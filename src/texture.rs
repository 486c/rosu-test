use std::path::Path;
use image::{io::Reader as ImageReader, GenericImageView, DynamicImage};
use wgpu::{ShaderStages, BindingType, TextureSampleType, TextureViewDimension};

use crate::graphics::Graphics;

pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl Texture {
    pub fn from_path<P: AsRef<Path>>(
        path: P, graphics: &Graphics
    ) -> Self {
        let image = ImageReader::open(path).unwrap()
            .decode().unwrap();

        Self::from_image(image, graphics)
    }

    pub fn from_image(image: DynamicImage, graphics: &Graphics) -> Self {
        let dimensions = image.dimensions();
        let buffer = image.to_rgba8();

        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };

        let texture = graphics.device.create_texture(
            &wgpu::TextureDescriptor {
                label: Some("Whatever"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            }
            );

        graphics.queue.write_texture(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &buffer,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1),
            },
            size,
            );

        let view = texture.create_view(
            &wgpu::TextureViewDescriptor::default()
        );

        let sampler = graphics.device.create_sampler(
            &wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            }
        );

        let bind_group_layout = graphics.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("texture_something"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float {
                                filterable: true
                            },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler (
                            wgpu::SamplerBindingType::Filtering
                        ),
                        count: None,
                    },
                ],
            });


        let bind_group = graphics.device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &view
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            &sampler
                        ),
                    },
                ],
                label: Some("hit_circle_bind"),
            }
        );

        Self {
            texture,
            view,
            sampler,
            bind_group_layout,
            bind_group,
        }
    }
}
