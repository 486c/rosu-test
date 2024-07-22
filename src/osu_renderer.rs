use std::{mem::size_of, sync::Arc};

use cgmath::Vector2;
use smallvec::SmallVec;
use wgpu::{
    util::DeviceExt, BindGroup, BindingType, BufferUsages, CommandEncoder, Extent3d,
    RenderPipeline, ShaderStages, TextureDescriptor, TextureDimension, TextureSampleType,
    TextureUsages, TextureView, TextureViewDimension,
};
use winit::dpi::PhysicalSize;

static SLIDER_SCALE: f32 = 2.0;

use crate::{
    camera::Camera,
    graphics::Graphics,
    hit_circle_instance::{ApproachCircleInstance, HitCircleInstance},
    hit_objects::{self, slider::SliderRender, Object, SLIDER_FADEOUT_TIME},
    math::lerp,
    slider_instance::SliderInstance,
    texture::{DepthTexture, Texture},
    vertex::Vertex,
};

macro_rules! buffer_write_or_init {
    ($queue:expr, $device:expr, $buffer:expr, $data:expr, $t: ty) => {{
        let data_len = $data.len() as u64;
        let buffer_bytes_size = $buffer.size();

        let buffer_len = buffer_bytes_size / size_of::<$t>() as u64;

        if data_len <= buffer_len {
            $queue.write_buffer(&$buffer, 0, bytemuck::cast_slice($data))
        } else {
            let buffer = $device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice($data),
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            });

            $buffer = buffer;
        }
    }};
}

const QUAD_INDECIES: &[u16] = &[0, 1, 2, 0, 2, 3];

const OSU_COORDS_WIDTH: f32 = 512.0;
const OSU_COORDS_HEIGHT: f32 = 384.0;

const OSU_PLAYFIELD_BORDER_TOP_PERCENT: f32 = 0.117;
const OSU_PLAYFIELD_BORDER_BOTTOM_PERCENT: f32 = 0.0834;

fn get_hitcircle_diameter(cs: f32) -> f32 {
    ((1.0 - 0.7 * (cs - 5.0) / 5.0) / 2.0) * 128.0 * 1.00041
}

fn calc_playfield_scale_factor(screen_w: f32, screen_h: f32) -> f32 {
    let top_border_size = OSU_PLAYFIELD_BORDER_TOP_PERCENT * screen_h;
    let bottom_border_size = OSU_PLAYFIELD_BORDER_BOTTOM_PERCENT * screen_h;

    let engine_screen_w = screen_w;
    let engine_screen_h = screen_h - bottom_border_size - top_border_size;

    let scale_factor = if screen_w / OSU_COORDS_WIDTH > engine_screen_h / OSU_COORDS_HEIGHT {
        engine_screen_h / OSU_COORDS_HEIGHT
    } else {
        engine_screen_w / OSU_COORDS_WIDTH
    };

    return scale_factor;
}

pub struct OsuRenderer {
    // Graphics State
    graphics: Graphics,

    // State
    scale: f32,
    offsets: Vector2<f32>,
    hit_circle_diameter: f32,

    // Quad verticies
    quad_verticies: [Vertex; 4],

    // Camera
    camera: Camera,
    camera_bind_group: BindGroup,
    camera_buffer: wgpu::Buffer,

    // Approach circle
    approach_circle_pipeline: RenderPipeline,
    approach_circle_texture: Texture,
    approach_circle_instance_buffer: wgpu::Buffer,
    approach_circle_instance_data: SmallVec<[ApproachCircleInstance; 32]>,

    // Hit Circle
    hit_circle_texture: Texture,
    hit_circle_pipeline: RenderPipeline,
    hit_circle_vertex_buffer: wgpu::Buffer,
    hit_circle_index_buffer: wgpu::Buffer,
    hit_circle_instance_data: Vec<HitCircleInstance>,
    hit_circle_instance_buffer: wgpu::Buffer,

    // Slider to texture
    slider_instance_buffer: wgpu::Buffer,
    slider_instance_data: Vec<SliderInstance>,
    slider_pipeline: RenderPipeline,
    slider_indecies: SmallVec<[u16; 16]>,

    slider_vertex_buffer: wgpu::Buffer,
    slider_index_buffer: wgpu::Buffer,
    slider_verticies: SmallVec<[Vertex; 256]>,

    // Slider texture to screen
    slider_to_screen_verticies: [Vertex; 4],
    slider_to_screen_vertex_buffer: wgpu::Buffer,
    slider_to_screen_render_pipeline: RenderPipeline,
    slider_to_screen_instance_buffer: wgpu::Buffer,
    slider_to_screen_instance_data: Vec<SliderInstance>,

    // Slider follow circle
    follow_point_texture: Texture,
    follow_points_instance_data: Vec<HitCircleInstance>,
    follow_points_instance_buffer: wgpu::Buffer,

    // Slider body queue
    slider_to_screen_textures: SmallVec<[(Arc<Texture>, Arc<wgpu::Buffer>, Option<u32>); 32]>,

    depth_texture: DepthTexture,

    prev_time: f64,
}

impl OsuRenderer {
    pub fn new(graphics: Graphics) -> Self {
        let hit_circle_texture = Texture::from_path("skin/hitcircle.png", &graphics);

        let approach_circle_texture = Texture::from_path("skin/approachcircle.png", &graphics);

        let follow_point_texture = Texture::from_path("skin/sliderb0.png", &graphics);

        let hit_circle_shader = graphics
            .device
            .create_shader_module(wgpu::include_wgsl!("shaders/hit_circle.wgsl"));

        let approach_circle_shader = graphics
            .device
            .create_shader_module(wgpu::include_wgsl!("shaders/approach_circle.wgsl"));

        let slider_shader = graphics
            .device
            .create_shader_module(wgpu::include_wgsl!("shaders/slider.wgsl"));

        let slider_to_screen_shader = graphics
            .device
            .create_shader_module(wgpu::include_wgsl!("shaders/slider_to_screen.wgsl"));

        let depth_texture =
            DepthTexture::new(&graphics, graphics.config.width, graphics.config.height, 1);

        let quad_verticies = Vertex::quad_centered(1.0, 1.0);

        let hit_circle_vertex_buffer =
            graphics
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("hit_circle_buffer"),
                    contents: bytemuck::cast_slice(&quad_verticies),
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                });

        let hit_circle_index_buffer =
            graphics
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("hit_circle_index_buffer"),
                    contents: bytemuck::cast_slice(QUAD_INDECIES),
                    usage: BufferUsages::INDEX,
                });

        let hit_circle_instance_data = Vec::new();

        let hit_circle_instance_buffer =
            graphics
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Hit Instance Buffer"),
                    contents: bytemuck::cast_slice(&hit_circle_instance_data),
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                });

        let approach_circle_instance_data = SmallVec::new();

        let approach_circle_instance_buffer =
            graphics
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Hit Instance Buffer"),
                    contents: bytemuck::cast_slice(&approach_circle_instance_data),
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                });

        /* Camera stuff */
        let camera = Camera::new(
            graphics.config.width as f32,
            graphics.config.height as f32,
            1.0,
        );

        let camera_buffer = graphics
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("uniform_buffer"),
                contents: bytemuck::bytes_of(&camera),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

        let camera_bind_group_layout =
            graphics
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                    label: Some("camera_bind_group_layout"),
                });

        let camera_bind_group = graphics
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &camera_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                }],
                label: Some("camera_bind_group"),
            });

        let approach_circle_pipeline_layout =
            graphics
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("approach circle pipeline Layout"),
                    bind_group_layouts: &[
                        &approach_circle_texture.bind_group_layout,
                        &camera_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });

        let approach_circle_pipeline =
            graphics
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("approach circle render pipeline"),
                    layout: Some(&approach_circle_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &approach_circle_shader,
                        entry_point: "vs_main",
                        buffers: &[Vertex::desc(), ApproachCircleInstance::desc()],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &approach_circle_shader,
                        entry_point: "fs_main",
                        targets: &[Some(wgpu::ColorTargetState {
                            format: graphics.config.format,
                            blend: Some(wgpu::BlendState {
                                color: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::SrcAlpha,
                                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                    operation: wgpu::BlendOperation::Add,
                                },
                                alpha: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::SrcAlpha,
                                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                    operation: wgpu::BlendOperation::Add,
                                },
                            }),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: Some(wgpu::Face::Back),
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: None/*Some(wgpu::DepthStencilState {
                        format: DepthTexture::DEPTH_FORMAT,
                        depth_write_enabled: false,
                        depth_compare: wgpu::CompareFunction::Always, // 1.
                        stencil: wgpu::StencilState::default(),     // 2.
                        bias: wgpu::DepthBiasState::default(),
                    })*/,
                    multisample: wgpu::MultisampleState {
                        count: 1,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview: None,
                });

        let hit_circle_pipeline_layout =
            graphics
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("hitcircle pipeline Layout"),
                    bind_group_layouts: &[
                        &hit_circle_texture.bind_group_layout,
                        &camera_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });

        let hit_circle_pipeline =
            graphics
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("hit_circle render pipeline"),
                    layout: Some(&hit_circle_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &hit_circle_shader,
                        entry_point: "vs_main",
                        buffers: &[Vertex::desc(), HitCircleInstance::desc()],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &hit_circle_shader,
                        entry_point: "fs_main",
                        targets: &[Some(wgpu::ColorTargetState {
                            format: graphics.config.format,
                            blend: Some(wgpu::BlendState {
                                color: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::SrcAlpha,
                                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                    operation: wgpu::BlendOperation::Add,
                                },
                                alpha: wgpu::BlendComponent::OVER,
                            }),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: Some(wgpu::Face::Back),
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: None,/*Some(wgpu::DepthStencilState {
                        format: DepthTexture::DEPTH_FORMAT,
                        depth_write_enabled: false,
                        depth_compare: wgpu::CompareFunction::Always, // 1.
                        stencil: wgpu::StencilState::default(),     // 2.
                        bias: wgpu::DepthBiasState {
                            ..Default::default()
                        },
                    })*/
                    multisample: wgpu::MultisampleState {
                        count: 1,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview: None,
                });

        let (slider_verticies, slider_indecies) = Vertex::cone(5.0);
        let slider_instance_data: Vec<SliderInstance> = Vec::with_capacity(10);

        let slider_vertex_buffer =
            graphics
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("hit_circle_buffer"),
                    contents: bytemuck::cast_slice(&slider_verticies),
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                });

        let slider_instance_buffer =
            graphics
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("linear instance buffer"),
                    contents: bytemuck::cast_slice(&slider_instance_data),
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                });

        let slider_index_buffer =
            graphics
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("hit_circle_index_buffer"),
                    contents: bytemuck::cast_slice(&slider_indecies),
                    usage: BufferUsages::INDEX,
                });

        let slider_pipeline_layout =
            graphics
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("slider test pipeline Layout"),
                    bind_group_layouts: &[&camera_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let slider_pipeline =
            graphics
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("slider test pipeline"),
                    layout: Some(&slider_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &slider_shader,
                        entry_point: "vs_main",
                        buffers: &[Vertex::desc(), SliderInstance::desc()],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &slider_shader,
                        entry_point: "fs_main",
                        targets: &[Some(wgpu::ColorTargetState {
                            format: graphics.config.format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: Some(wgpu::Face::Back),
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: DepthTexture::DEPTH_FORMAT,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Less, // 1.
                        stencil: wgpu::StencilState::default(),     // 2.
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        ..Default::default()
                    },
                    multiview: None,
                });

        let slider_to_screen_verticies = Vertex::quad_positional(0.0, 0.0, 1.0, 1.0);

        let slider_to_screen_vertex_buffer =
            graphics
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("hit_circle_buffer"),
                    contents: bytemuck::cast_slice(&slider_to_screen_verticies),
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                });

        let slider_to_screen_bind_group_layout =
            graphics
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("slider to screen bind group layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::FRAGMENT,
                            ty: BindingType::Texture {
                                sample_type: TextureSampleType::Float { filterable: true },
                                view_dimension: TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::FRAGMENT,
                            ty: BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

        let slider_to_screen_pipeline_layout =
            graphics
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("slider to screen pipeline Layout"),
                    bind_group_layouts: &[
                        //&Texture::default_bind_group_layout(&graphics, 1),
                        &slider_to_screen_bind_group_layout,
                        &camera_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });

        let slider_to_screen_render_pipeline =
            graphics
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("slider to screen render pipeline23"),
                    layout: Some(&slider_to_screen_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &slider_to_screen_shader,
                        entry_point: "vs_main",
                        buffers: &[Vertex::desc(), SliderInstance::desc()],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &slider_to_screen_shader,
                        entry_point: "fs_main",
                        targets: &[Some(wgpu::ColorTargetState {
                            format: graphics.config.format,
                            blend: Some(wgpu::BlendState {
                                color: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::SrcAlpha,
                                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                    operation: wgpu::BlendOperation::Add,
                                },
                                alpha: wgpu::BlendComponent::REPLACE,
                            }),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: Some(wgpu::Face::Back),
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: DepthTexture::DEPTH_FORMAT,
                        depth_write_enabled: false,
                        depth_compare: wgpu::CompareFunction::Always, // 1.
                        stencil: wgpu::StencilState::default(),     // 2.
                        bias: wgpu::DepthBiasState {
                            clamp: 1.0,
                            ..Default::default()
                        },
                    }),
                    multisample: wgpu::MultisampleState {
                        count: 1,
                        ..Default::default()
                    },
                    multiview: None,
                });

        let slider_to_screen_instance_data = Vec::with_capacity(10);

        let slider_to_screen_instance_buffer =
            graphics
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("slider to screen instance buffer"),
                    contents: bytemuck::cast_slice(&slider_instance_data),
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                });

        let follow_points_instance_data = Vec::with_capacity(10);

        let follow_points_instance_buffer =
            graphics
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("slider to screen instance buffer"),
                    contents: bytemuck::cast_slice(&follow_points_instance_data),
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                });

        let scale =
            calc_playfield_scale_factor(graphics.size.width as f32, graphics.size.height as f32);

        Self {
            graphics,
            scale,
            quad_verticies,
            camera,
            camera_bind_group,
            camera_buffer,
            approach_circle_pipeline,
            approach_circle_texture,
            approach_circle_instance_buffer,
            approach_circle_instance_data,
            hit_circle_texture,
            hit_circle_pipeline,
            hit_circle_vertex_buffer,
            hit_circle_index_buffer,
            hit_circle_instance_data,
            hit_circle_instance_buffer,
            depth_texture,
            slider_instance_buffer,
            slider_instance_data,
            slider_pipeline,
            slider_indecies: slider_indecies.into(),
            slider_vertex_buffer,
            slider_index_buffer,
            slider_verticies: slider_verticies.into(),
            slider_to_screen_verticies,
            slider_to_screen_vertex_buffer,
            slider_to_screen_render_pipeline,
            slider_to_screen_instance_buffer,
            slider_to_screen_instance_data,
            slider_to_screen_textures: SmallVec::new(),
            follow_point_texture,
            follow_points_instance_data,
            follow_points_instance_buffer,
            offsets: Vector2::new(0.0, 0.0),
            hit_circle_diameter: 1.0,
            prev_time: 0.0,
        }
    }

    pub fn prepare_objects2(
        &mut self, 
        time: f64,
        preempt: f32,
        fadein: f32,
        queue: &[usize], 
        objects: &[Object]
    ) {
        let _span = tracy_client::span!("osu_renderer prepare_objects2");

        // Calculating Z values for current queue
        let total = queue.len() as f32;
        let step = 1.0 / total;
        let mut curr_val = 0.0;

        for current_index in queue.iter().rev() {
            assert!(curr_val <= 1.0);

            //println!("z val: {}", curr_val);

            let object = &objects[*current_index];

            match &object.kind {
                hit_objects::ObjectKind::Circle(circle) => {
                    let _span = tracy_client::span!("osu_renderer prepare_objects2::circle");
                    let start_time = object.start_time - preempt as f64;
                    let end_time = start_time + fadein as f64;
                    let alpha = ((time - start_time) / (end_time - start_time)).clamp(0.0, 1.0);

                    let approach_progress = (time - start_time) / (object.start_time - start_time);

                    let approach_scale = lerp(1.0, 4.0, 1.0 - approach_progress).clamp(1.0, 4.0);

                    self.hit_circle_instance_data.push(HitCircleInstance::new(
                            circle.pos.x,
                            circle.pos.y,
                            curr_val,
                            alpha as f32,
                    ));

                    self.approach_circle_instance_data
                        .push(ApproachCircleInstance::new(
                                circle.pos.x,
                                circle.pos.y,
                                curr_val,
                                alpha as f32,
                                approach_scale as f32,
                        ));
                },
                hit_objects::ObjectKind::Slider(slider) => {
                    let _span = tracy_client::span!("osu_renderer prepare_objects2::circle");

                    let start_time = slider.start_time - preempt as f64;
                    let end_time = start_time + fadein as f64;

                    let mut body_alpha =
                        ((time - start_time) / (end_time - start_time)).clamp(0.0, 0.95);

                    // FADEOUT
                    if time >= object.start_time + slider.duration
                        && time <= object.start_time + slider.duration + SLIDER_FADEOUT_TIME
                    {
                        let start = object.start_time + slider.duration;
                        let end = object.start_time + slider.duration + SLIDER_FADEOUT_TIME;

                        let min = start.min(end);
                        let max = start.max(end);

                        let percentage = 100.0 - (((time - min) * 100.0) / (max - min)); // TODO remove `* 100.0`

                        body_alpha = (percentage / 100.0).clamp(0.0, 0.95);
                    }

                    // APPROACH
                    let approach_progress = (time - start_time) / (object.start_time - start_time);

                    let approach_scale = lerp(1.0, 3.95, 1.0 - approach_progress).clamp(1.0, 4.0);

                    let approach_alpha = if time >= object.start_time {
                        0.0
                    } else {
                        body_alpha
                    };

                    // FOLLOW CIRCLE STUFF
                    // BLOCK IN WHICH SLIDER IS HITABLE
                    let mut follow_circle = None;
                    if time >= object.start_time && time <= object.start_time + slider.duration {
                        // Calculating current slide
                        let v1 = time - object.start_time;
                        let v2 = slider.duration / slider.repeats as f64;
                        let slide = (v1 / v2).floor() as i32 + 1;

                        let slide_start = object.start_time + (v2 * (slide as f64 - 1.0));

                        let start = slide_start;
                        let current = time;
                        let end = slide_start + v2;

                        let min = start.min(end);
                        let max = start.max(end);

                        let mut percentage = ((current - min) * 100.0) / (max - min); // TODO remove `* 100.0`

                        // If slide is even we should go from 100% to 0%
                        // if not then from 0% to 100%
                        if slide % 2 == 0 {
                            percentage = 100.0 - percentage;
                        }

                        let pos = slider.curve.position_at(percentage / 100.0);

                        self.follow_points_instance_data.push(HitCircleInstance {
                            pos: [pos.x + slider.pos.x, pos.y + slider.pos.y, curr_val],
                            alpha: body_alpha as f32,
                        });

                        follow_circle = Some(self.follow_points_instance_data.len() as u32);
                    }

                    // BODY
                    self.slider_to_screen_instance_data.push(SliderInstance {
                        pos: [0.0, 0.0, curr_val],
                        alpha: body_alpha as f32,
                    });

                    self.approach_circle_instance_data
                        .push(ApproachCircleInstance::new(
                                slider.pos.x,
                                slider.pos.y,
                                curr_val,
                                approach_alpha as f32,
                                approach_scale as f32,
                        ));

                    // That's tricky part. Since every slider have a according
                    // slider texture and a quad where texture will be rendered and presented on screen.
                    // So we are pushing all textures to the "queue" so we can iterate on it later
                    if let Some(render) = &slider.render {
                        self.slider_to_screen_textures.push((
                                render.texture.clone(),
                                render.quad.clone(),
                                follow_circle,
                        ))
                    } else {
                        panic!("Texture and quad should be present");
                    };

                },
            }

            curr_val += step;
        }

    }

    pub fn get_graphics(&self) -> &Graphics {
        &self.graphics
    }

    /// Render slider to the **texture** not screen
    pub fn prepare_and_render_slider_texture(
        &mut self,
        slider: &mut crate::hit_objects::slider::Slider,
    ) {
        // TODO optimization idea

        let _span = tracy_client::span!("osu_renderer prepare_and_render_slider_texture");

        if !slider.render.is_none() {
            return;
        }

        let bbox = slider.bounding_box((self.hit_circle_diameter / 2.0) * SLIDER_SCALE);

        let (slider_vertices, _) = Vertex::cone((self.hit_circle_diameter / 2.0) * SLIDER_SCALE);

        self.slider_verticies = slider_vertices.into();

        buffer_write_or_init!(
            self.graphics.queue,
            self.graphics.device,
            self.slider_vertex_buffer,
            &self.slider_verticies,
            Vertex
        );
        let slider_bounding_box = bbox.clone();

        let bbox_width = bbox.width() * 2.0;
        let bbox_height = bbox.height() * 2.0;

        let depth_texture =
            DepthTexture::new(&self.graphics, bbox_width as u32, bbox_height as u32, 1);

        let ortho = Camera::ortho(0.0, bbox_width, bbox_height, 0.0);

        let camera_buffer =
            self.graphics
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("uniform_buffer"),
                    contents: bytemuck::bytes_of(&ortho),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                });

        let camera_bind_group_layout =
            self.graphics
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                    label: Some("camera_bind_group_layout"),
                });

        let camera_bind_group =
            self.graphics
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &camera_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buffer.as_entire_binding(),
                    }],
                    label: Some("camera_bind_group"),
                });

        self.slider_instance_data.clear();

        let slider_texture_not_sampled = self.graphics.device.create_texture(&TextureDescriptor {
            label: Some("SLIDER RENDER TEXTURE"),
            size: Extent3d {
                width: bbox_width as u32,
                height: bbox_height as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: self.graphics.config.format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[self.graphics.config.format],
        });

        // Preparing instances
        let curve = &slider.curve;
        let n_segments = curve.dist() / 2.5;
        let step_by = (100.0 / n_segments as f64) / 100.0;

        let mut step = 0.0;

        while step <= 1.0 {
            let p = curve.position_at(step);

            // translating a bounding box coordinates to our coordinates that starts at (0,0)
            let x = p.x + slider.pos.x;
            let x = 0.0 + (x - bbox.top_left.x);

            let y = p.y + slider.pos.y;
            let y = 0.0 + (y - bbox.top_left.y);

            self.slider_instance_data.push(SliderInstance::new(
                x * SLIDER_SCALE,
                y * SLIDER_SCALE,
                0.0,
                1.0,
            ));

            step += step_by;
        }

        let mut origin = Vector2::new(slider.pos.x, slider.pos.y);
        origin.x = 0.0 + (origin.x - bbox.top_left.x);
        origin.y = 0.0 + (origin.y - bbox.top_left.y);

        buffer_write_or_init!(
            self.graphics.queue,
            self.graphics.device,
            self.slider_instance_buffer,
            &self.slider_instance_data,
            SliderInstance
        );

        /*
        self.slider_instance_buffer =
            self.graphics
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("linear vertex_buffer"),
                    contents: bytemuck::cast_slice(&self.slider_instance_data),
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                });
                */

        // Drawing to the texture
        let view = slider_texture_not_sampled.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            self.graphics
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("SLIDER TEXTURE ENCODER"),
                });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("slider render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.slider_pipeline);

            render_pass.set_bind_group(0, &camera_bind_group, &[]);

            render_pass.set_vertex_buffer(0, self.slider_vertex_buffer.slice(..));

            render_pass.set_vertex_buffer(1, self.slider_instance_buffer.slice(..));

            render_pass.set_index_buffer(
                self.slider_index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );

            render_pass.draw_indexed(
                0..self.slider_indecies.len() as u32,
                0,
                0..self.slider_instance_data.len() as u32,
            );
        }

        self.graphics
            .queue
            .submit(std::iter::once(encoder.finish()));

        let slider_texture = Arc::new(Texture::from_texture(
            slider_texture_not_sampled,
            &self.graphics,
            1,
        ));

        // RENDERED SLIDER TEXTURE QUAD
        let verticies = Vertex::quad_origin(
            slider.pos.x - origin.x,
            slider.pos.y - origin.y,
            bbox_width / SLIDER_SCALE,
            bbox_height / SLIDER_SCALE,
        );

        let slider_quad =
            self.graphics
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("another slider to screen verticies buffer"),
                    contents: bytemuck::cast_slice(&verticies),
                    usage: BufferUsages::VERTEX,
                });

        slider.render = Some(SliderRender {
            texture: slider_texture,
            quad: slider_quad.into(),
            bounding_box: slider_bounding_box,
        });
    }

    pub fn on_cs_change(&mut self, cs: f32) {
        println!("OsuRenderer -> on_cs_change()");
        let hit_circle_diameter = get_hitcircle_diameter(cs);

        self.hit_circle_diameter = hit_circle_diameter;

        self.quad_verticies = Vertex::quad_centered(hit_circle_diameter, hit_circle_diameter);

        // TODO temp

        self.hit_circle_vertex_buffer =
            self.graphics
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("hit_circle_buffer"),
                    contents: bytemuck::cast_slice(&self.quad_verticies),
                    usage: BufferUsages::VERTEX,
                });

        // Slider
        let (slider_vertices, slider_index) = Vertex::cone(hit_circle_diameter / 2.0);

        self.slider_verticies = slider_vertices.into();

        buffer_write_or_init!(
            self.graphics.queue,
            self.graphics.device,
            self.slider_vertex_buffer,
            &self.slider_verticies,
            Vertex
        );

        self.slider_indecies = slider_index.into();

        self.slider_index_buffer =
            self.graphics
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("hit_circle_buffer"),
                    contents: bytemuck::cast_slice(&self.slider_indecies),
                    usage: BufferUsages::INDEX,
                });
    }

    pub fn on_resize(&mut self, new_size: &PhysicalSize<u32>) {
        self.graphics.resize(new_size);

        // Calculate playfield scale
        self.scale = calc_playfield_scale_factor(new_size.width as f32, new_size.height as f32);

        // Calculate playfield offsets
        let scaled_height = OSU_COORDS_HEIGHT as f32 * self.scale;
        let scaled_width = OSU_COORDS_WIDTH as f32 * self.scale;

        let bottom_border_size = OSU_PLAYFIELD_BORDER_BOTTOM_PERCENT * new_size.height as f32;

        let y_offset = (new_size.height as f32 - scaled_height) / 2.0
            + (new_size.height as f32 / 2.0 - (scaled_height / 2.0) - bottom_border_size);

        let x_offset = (new_size.width as f32 - scaled_width) / 2.0;

        let offsets = Vector2::new(x_offset, y_offset);
        self.offsets = offsets;

        self.camera.resize(new_size);
        self.camera.transform(self.scale, offsets);
        self.depth_texture = DepthTexture::new(
            &self.graphics,
            self.graphics.config.width,
            self.graphics.config.height,
            1,
        );

        // TODO Recreate buffers
        self.graphics
            .queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&self.camera)); // TODO

        // Slider to screen

        self.slider_to_screen_verticies = Vertex::quad_positional(
            0.0,
            0.0,
            self.graphics.config.width as f32,
            self.graphics.config.height as f32,
        );

        buffer_write_or_init!(
            self.graphics.queue,
            self.graphics.device,
            self.slider_to_screen_vertex_buffer,
            &self.slider_to_screen_verticies,
            Vertex
        );
    }

    pub fn write_buffers(&mut self) {
        let _span = tracy_client::span!("osu_renderer write buffers");
        //println!("==============");
        
        /*
        // TODO remove later
        let total = self.hit_circle_instance_data.len() as f32;
        //let step = 1.0 / ((1.0 - 0.0) / (total - 1.0));
        let step = 2.0 / total;
        let mut curr_val = -1.0;
        for hitobject in &mut self.hit_circle_instance_data {
            assert!(curr_val <= 1.0);
            hitobject.pos[2] = curr_val;
            println!("z: {curr_val}: step: {step}: total: {total}");
            curr_val += step;
        }
        */

        //self.hit_circle_instance_data = self.hit_circle_instance_data.clone().into_iter().rev().collect();

        buffer_write_or_init!(
            self.graphics.queue,
            self.graphics.device,
            self.hit_circle_instance_buffer,
            &self.hit_circle_instance_data,
            HitCircleInstance
        );
        
        buffer_write_or_init!(
            self.graphics.queue,
            self.graphics.device,
            self.approach_circle_instance_buffer,
            &self.approach_circle_instance_data,
            ApproachCircleInstance
        );

        buffer_write_or_init!(
            self.graphics.queue,
            self.graphics.device,
            self.slider_to_screen_instance_buffer,
            &self.slider_to_screen_instance_data,
            SliderInstance
        );

        buffer_write_or_init!(
            self.graphics.queue,
            self.graphics.device,
            self.follow_points_instance_buffer,
            &self.follow_points_instance_data,
            SliderInstance
        );
    }

    /// Clears internal buffers
    pub fn clear_buffers(&mut self) {
        let _span = tracy_client::span!("osu_renderer clear_buffers");
        self.hit_circle_instance_data.clear();
        self.approach_circle_instance_data.clear();
        self.slider_to_screen_instance_data.clear();
        self.slider_to_screen_textures.clear();
        self.follow_points_instance_data.clear();
    }

    /// Prepares object for render
    /// HitCircle:
    ///     1. Prepare hit & approach circles instances
    /// Slider:
    ///
    pub fn prepare_object_for_render(
        &mut self,
        obj: &Object,
        time: f64,
        preempt: f32,
        fadein: f32,
    ) {
        match &obj.kind {
            hit_objects::ObjectKind::Circle(circle) => {
                let _span = tracy_client::span!("osu_renderer prepare_object_for_render::circle");
                
                /*
                let start_time = obj.start_time - preempt as f64;
                let end_time = start_time + fadein as f64;
                let alpha = ((time - start_time) / (end_time - start_time)).clamp(0.0, 1.0);

                let approach_progress = (time - start_time) / (obj.start_time - start_time);

                let approach_scale = lerp(1.0, 4.0, 1.0 - approach_progress).clamp(1.0, 4.0);

                self.hit_circle_instance_data.push(HitCircleInstance::new(
                    circle.pos.x,
                    circle.pos.y,
                    0.0,
                    alpha as f32,
                ));

                self.approach_circle_instance_data
                    .push(ApproachCircleInstance::new(
                        circle.pos.x,
                        circle.pos.y,
                        alpha as f32,
                        approach_scale as f32,
                    ));
                */
            }
            hit_objects::ObjectKind::Slider(slider) => {
            }
        }
    }

    /// Render all sliders from the queue
    pub fn render_sliders(
        &mut self,
        view: &TextureView,
    ) -> Result<CommandEncoder, wgpu::SurfaceError> {
        let _span = tracy_client::span!("osu_renderer render_sliders");

        let mut encoder =
            self.graphics
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("sliders & followpoints encoder"),
                });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("slider render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Sanity check
            assert_eq!(
                self.slider_to_screen_instance_data.len(),
                self.slider_to_screen_textures.len()
            );

            render_pass.set_pipeline(&self.slider_to_screen_render_pipeline);
            render_pass.set_vertex_buffer(1, self.slider_to_screen_instance_buffer.slice(..));
            render_pass.set_index_buffer(
                self.hit_circle_index_buffer.slice(..), // DOCS
                wgpu::IndexFormat::Uint16,
            );


            for (i, (texture, vertex_buffer, _follow)) in
                self.slider_to_screen_textures.iter().enumerate().rev()
            {
                let instance = i as u32..i as u32 + 1;

                render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));

                render_pass.set_bind_group(0, &texture.bind_group, &[]);

                // First draw a slider body
                render_pass.draw_indexed(0..QUAD_INDECIES.len() as u32, 0, instance);
            }

            render_pass.set_pipeline(&self.hit_circle_pipeline);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(0, &self.follow_point_texture.bind_group, &[]);

            render_pass.set_vertex_buffer(0, self.hit_circle_vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.follow_points_instance_buffer.slice(..));
            render_pass.set_index_buffer(
                self.hit_circle_index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );

            render_pass.draw_indexed(
                0..QUAD_INDECIES.len() as u32, 
                0, 
                0..self.follow_points_instance_data.len() as u32
            );
        }

        Ok(encoder)
    }

    pub fn render_hitcircles(
        &mut self,
        view: &TextureView,
    ) -> Result<CommandEncoder, wgpu::SurfaceError> {
        let _span = tracy_client::span!("osu_renderer render_hitcircles");
        let mut encoder =
            self.graphics
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("HitCircles encoder"),
                });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("slider render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        /*
                        wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                        }),
                        */
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None/*Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                })*/,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // HIT CIRCLES
            render_pass.set_pipeline(&self.hit_circle_pipeline);
            render_pass.set_bind_group(0, &self.hit_circle_texture.bind_group, &[]);

            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);

            render_pass.set_vertex_buffer(0, self.hit_circle_vertex_buffer.slice(..));

            render_pass.set_vertex_buffer(1, self.hit_circle_instance_buffer.slice(..));

            render_pass.set_index_buffer(
                self.hit_circle_index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );

            render_pass.draw_indexed(
                0..QUAD_INDECIES.len() as u32,
                0,
                0..self.hit_circle_instance_data.len() as u32,
            );

            // APPROACH CIRCLES
            render_pass.set_pipeline(&self.approach_circle_pipeline);
            render_pass.set_bind_group(
                // TODO ???
                0,
                &self.approach_circle_texture.bind_group,
                &[],
            );

            render_pass.set_bind_group(1, &self.approach_circle_texture.bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);

            render_pass.set_vertex_buffer(1, self.approach_circle_instance_buffer.slice(..));

            render_pass.draw_indexed(
                0..QUAD_INDECIES.len() as u32,
                0,
                0..self.approach_circle_instance_data.len() as u32,
            );
        }

        Ok(encoder)
    }

    /// Render all objects from internal buffers
    /// and clears used buffers afterwards
    pub fn render_objects(&mut self, view: &TextureView) -> Result<(), wgpu::SurfaceError> {
        let _span = tracy_client::span!("osu_renderer render_objects");

        let hitcircles_encoder = self.render_hitcircles(&view)?;
        //let sliders_encoder = self.render_sliders(&view)?;

        let span = tracy_client::span!("osu_renderer render_objects::queue::submit");
        self.graphics
            .queue
            .submit([hitcircles_encoder.finish()]);
        drop(span);

        self.clear_buffers();

        Ok(())
    }
}
