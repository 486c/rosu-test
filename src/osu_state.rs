use std::{fs::File, io::BufReader, path::{Path, PathBuf}, sync::{mpsc::{channel, Receiver, TryRecvError}, Arc}, time::Duration};

use egui::Slider;
use egui_file::FileDialog;
use rodio::{Decoder, Sink};
use rosu_map::Beatmap;
use winit::{dpi::PhysicalSize, window::Window};

use crate::{
    config::Config, egui_state::EguiState, graphics::Graphics, hit_objects::{Object, ObjectKind}, osu_renderer::OsuRenderer, skin_manager::SkinManager, timer::Timer, ui::settings::SettingsView
};

pub enum OsuStateEvent {
    ChangeSkin(PathBuf),
}

/// Return preempt and fadein based on AR
fn calculate_preempt_fadein(ar: f32) -> (f32, f32) {
    if ar > 5.0 {
        (
            1200.0 - 750.0 * (ar - 5.0) / 5.0,
            800.0 - 500.0 * (ar - 5.0) / 5.0,
        )
    } else if ar < 5.0 {
        (
            1200.0 + 600.0 * (5.0 - ar) / 5.0,
            800.0 + 400.0 * (5.0 - ar) / 5.0,
        )
    } else {
        (1200.0, 800.0)
    }
}

fn calculate_hit_window(od: f32) -> (f32, f32, f32) {
    (80.0 - 6.0 * od, 140.0 - 8.0 * od, 200.0 - 10.0 * od)
}

pub struct OsuState<'s> {
    pub window: Arc<Window>,
    pub egui: EguiState,
    pub event_receiver: Receiver<OsuStateEvent>,

    pub sink: Sink,

    skin_manager: SkinManager,
    config: Config,
    settings_view: SettingsView,

    osu_renderer: OsuRenderer<'s>,

    current_beatmap: Option<Beatmap>,
    preempt: f32,
    fadein: f32,
    hit_offset: f32,

    hit_objects: Vec<Object>,
    objects_queue: Vec<usize>,

    osu_clock: Timer,
    
    // I hate that i have to store it right here, but i'm gonna leave it here
    // just for easier debugging and prototyping
    file_dialog: Option<FileDialog>,
    difficulties: Vec<PathBuf>,
    new_beatmap: Option<PathBuf>,
}

impl<'s> OsuState<'s> {
    pub fn new(window: Arc<Window>, graphics: Graphics<'s>, sink: Sink) -> Self {
        let egui = EguiState::new(&graphics, &window);
        let skin_manager = SkinManager::from_path("skin", &graphics);
        let config = Config::default();
        let osu_renderer = OsuRenderer::new(graphics, &config);

        let (tx, event_receiver) = channel::<OsuStateEvent>();

        Self {
            event_receiver,
            preempt: 0.0,
            hit_offset: 0.0,
            fadein: 0.0,
            osu_renderer,
            window,
            current_beatmap: None,
            egui,
            sink,
            osu_clock: Timer::new(),
            objects_queue: Vec::with_capacity(20),
            hit_objects: Vec::new(),
            file_dialog: None,
            difficulties: Vec::new(),
            new_beatmap: None,
            skin_manager,
            config,
            settings_view: SettingsView::new(tx.clone()),
        }
    }

    pub fn open_skin(&mut self, path: impl AsRef<Path>) {
        let skin_manager = SkinManager::from_path(path, self.osu_renderer.get_graphics());
        self.skin_manager = skin_manager;
    }

    pub fn open_beatmap(&mut self, path: impl AsRef<Path>) {
        self.osu_clock.reset_time();

        let map = match Beatmap::from_path(path.as_ref()) {
            Ok(m) => m,
            Err(e) => {
                println!("Failed to parse beatmap");
                println!("{}", e);
                return;
            }
        };

        // Prepare audio
        self.sink.clear();

        let beatmap_dir = path.as_ref().parent().expect("failed to get beatmap dir");
        let audio_file = beatmap_dir.join(&map.audio_file);
        
        // We have to acknowlage the fact that there might be beatmaps
        // without any audio files
        if audio_file.is_file() {
            let file = BufReader::new(File::open(audio_file).unwrap());
            let source = Decoder::new(file).expect("Failed to load audio file source");
            self.sink.append(source);
            println!("open_beatmap: Initialized a new audio file!");
        }

        let (preempt, fadein) = calculate_preempt_fadein(map.approach_rate);
        let (_x300, _x100, x50) = calculate_hit_window(map.overall_difficulty);

        self.preempt = preempt;
        self.fadein = fadein;
        self.hit_offset = x50;

        // Convert rosu_map object to our objects
        //let mut out_objects = Vec::with_capacity(map.hit_objects.len());
        let out_objects = Object::from_rosu(&map.hit_objects);
        /*
        for obj in &map.hit_objects {
            if let Some(cobj) = Object::from_rosu(obj) {
                out_objects.push(cobj)
            }
        }
        */
        self.hit_objects = out_objects;

        self.current_beatmap = Some(map);
        self.apply_beatmap_transformations();
    }

    pub fn apply_beatmap_transformations(&mut self) {
        let cs = match &self.current_beatmap {
            Some(beatmap) => beatmap.circle_size,
            None => 4.0,
        };

        self.osu_renderer.on_cs_change(cs);
    }

    pub fn resize(&mut self, new_size: &PhysicalSize<u32>) {
        self.osu_renderer.on_resize(new_size);
    }

    pub fn update_egui(&mut self) {
        let _span = tracy_client::span!("osu_state update egui");

        let input = self.egui.state.take_egui_input(&self.window);

        self.egui.state.egui_ctx().begin_frame(input);

        self.settings_view.window(
            &self.egui.state.egui_ctx(),
            &self.skin_manager,
            &mut self.config,
        );

        egui::Window::new("Window").show(&self.egui.state.egui_ctx(), |ui| {

            if ui.add(egui::Button::new("Select Beatmap")).clicked() {
                let mut dialog = FileDialog::select_folder(None);
                dialog.open();
                self.file_dialog = Some(dialog);
            }

            if let Some(dialog) = &mut self.file_dialog {
                if dialog.show(self.egui.state.egui_ctx()).selected() {
                    let mut available_choices = Vec::new();
                    if let Some(dir) = dialog.path() {
                        for entry in std::fs::read_dir(dir).expect("failed to read dir") {
                            let entry = entry.expect("failed to read dir entry");
                            if let Some(ext) = entry.path().extension() {
                                if ext == "osu" {
                                    available_choices.push(entry.path());
                                }
                            }
                        }

                        self.difficulties = available_choices;
                    }
                }
            }

            if let Some(beatmap) = &self.current_beatmap {
                ui.add(egui::Label::new(format!("{}", self.osu_clock.get_time())));

                if ui.add(
                    Slider::new(
                        &mut self.osu_clock.last_time,
                        1.0..=(beatmap.hit_objects.last().unwrap().start_time),
                    )
                    .step_by(1.0),
                ).changed() {
                    self.sink.try_seek(Duration::from_millis(self.osu_clock.get_time().round() as u64)).unwrap();
                };

                if !self.osu_clock.is_paused() {
                    if ui.add(egui::Button::new("pause")).clicked() {
                        self.osu_clock.pause();
                        self.sink.pause();
                    }
                } else {
                    if ui.add(egui::Button::new("unpause")).clicked() {
                        self.sink.try_seek(Duration::from_millis(self.osu_clock.get_time().round() as u64)).unwrap();
                        self.osu_clock.unpause();
                        self.sink.play();
                    }
                }
            }
        });

        if !self.difficulties.is_empty() {
            egui::Window::new("Select Difficulty").show(&self.egui.state.egui_ctx(), |ui| {
                for path in &self.difficulties {
                    if ui.add(egui::Button::new(format!("{:#?}", path.file_name().unwrap()))).clicked() {
                        self.new_beatmap = Some(path.to_path_buf());
                    };
                }

            });

        }

        let output = self.egui.state.egui_ctx().end_frame();

        self.egui.state.handle_platform_output(
            &self.window,
            output.platform_output.to_owned(),
        );

        self.egui.output = Some(output);
    }

    // Going through every object on beatmap and preparing it to
    // assigned buffers
    pub fn prepare_objects(&mut self, time: f64) {
        let _span = tracy_client::span!("osu_state prepare objects");

        for (i, obj) in self.hit_objects.iter_mut().enumerate().rev() {
            if !obj.is_visible(time, self.preempt) {
                continue;
            }

            // TODO circles
            match &mut obj.kind {
                ObjectKind::Circle(_) => {}
                ObjectKind::Slider(slider) => {
                    self.osu_renderer.prepare_and_render_slider_texture(slider, &self.skin_manager, &self.config);
                }
            }

            self.objects_queue.push(i);

            //self.osu_renderer
                //.prepare_object_for_render(obj, time, self.preempt, self.fadein);
        }

        self.osu_renderer.prepare_objects(
            time, self.preempt, self.fadein,
            &self.objects_queue, &self.hit_objects,
            &self.skin_manager
        );

        // Syncing osu state settings with the osu renderer
        self.osu_renderer.prepare(
            &self.config
        );

        // When we are done preparing all objects for rendering
        // we should not forget to upload all that to gpu
        self.osu_renderer.write_buffers();
    }

    pub fn update(&mut self) {
        let _span = tracy_client::span!("osu_state update");

        let event = self.event_receiver.try_recv();

        match event {
            Ok(event) => {
                match event {
                    OsuStateEvent::ChangeSkin(path) => {
                        self.open_skin(path)
                    },
                }
            },
            Err(TryRecvError::Empty) => {},
            _ => panic!("sender disconnected"),
        }

        if let Some(path) = &self.new_beatmap.clone() {
            self.open_beatmap(path);
            self.new_beatmap = None;
            self.difficulties.clear();
        }

        self.update_egui();
        let time = self.osu_clock.update();

        self.prepare_objects(time);
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let _span = tracy_client::span!("osu_state render");

        let span = tracy_client::span!("osu_state render::get_current_texture");
        let output = self.osu_renderer.get_graphics().get_current_texture()?;
        drop(span);

        let span = tracy_client::span!("osu_state render::create_view");
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        drop(span);

        self.osu_renderer.render_objects(
            &view,
            &self.objects_queue, &self.hit_objects,
            &self.skin_manager,
        )?;

        let graphics = self.osu_renderer.get_graphics();

        let mut encoder = graphics
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        self.egui.render(graphics, &mut encoder, &view)?;

        let span = tracy_client::span!("osu_state queue::submit");
        graphics.queue.submit(std::iter::once(encoder.finish()));
        drop(span);

        let span = tracy_client::span!("osu_state render::present");
        output.present();
        drop(span);

        // Clearing objects queue only after they successfully rendered
        self.objects_queue.clear();


        Ok(())
    }
}
