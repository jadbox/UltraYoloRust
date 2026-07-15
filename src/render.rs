use anyhow::{bail, Result};
use wgpu::util::DeviceExt;

use crate::types::{PoseDetection, COCO_SKELETON};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
}

struct FrameResources {
    width: u32,
    height: u32,
    source: wgpu::Texture,
    target: wgpu::Texture,
    target_view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    readback: wgpu::Buffer,
    padded_bytes_per_row: u32,
}

pub struct PoseRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    frame_resources: Option<FrameResources>,
}

impl PoseRenderer {
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .ok_or_else(|| anyhow::anyhow!("no GPU adapter available"))?;
        println!("GPU overlay renderer: {}", adapter.get_info().name);
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("pose renderer"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pose overlay shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("pose.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("source frame layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pose overlay pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pose overlay pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            frame_resources: None,
        })
    }

    fn ensure_frame_resources(&mut self, width: u32, height: u32) {
        if self
            .frame_resources
            .as_ref()
            .is_some_and(|r| r.width == width && r.height == height)
        {
            return;
        }
        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let source = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("source frame"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let target = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("annotated frame"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let source_view = source.create_view(&Default::default());
        let target_view = target.create_view(&Default::default());
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("source frame bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let padded_bytes_per_row = (width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pose readback"),
            size: (padded_bytes_per_row * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        self.frame_resources = Some(FrameResources {
            width,
            height,
            source,
            target,
            target_view,
            bind_group,
            readback,
            padded_bytes_per_row,
        });
    }

    fn render_inner(
        &mut self,
        rgba: &[u8],
        width: u32,
        height: u32,
        detections: &[PoseDetection],
        kpt_conf: f32,
        output: &mut Vec<u8>,
    ) -> Result<()> {
        if rgba.len() != width as usize * height as usize * 4 {
            bail!(
                "RGBA frame has {} bytes; expected {}",
                rgba.len(),
                width as usize * height as usize * 4
            );
        }
        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        self.ensure_frame_resources(width, height);
        let resources = self.frame_resources.as_ref().unwrap();
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &resources.source,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            extent,
        );
        let vertices = vertices(detections, width as f32, height as f32, kpt_conf);
        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pose vertices"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pose render encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pose render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &resources.target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &resources.bind_group, &[]);
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.draw(0..vertices.len() as u32, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &resources.target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &resources.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(resources.padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            extent,
        );
        self.queue.submit(Some(encoder.finish()));
        let slice = resources.readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|_| anyhow::anyhow!("GPU readback callback dropped"))??;
        let mapped = slice.get_mapped_range();
        output.resize((width * height * 4) as usize, 0);
        for y in 0..height as usize {
            let src = &mapped[y * resources.padded_bytes_per_row as usize
                ..y * resources.padded_bytes_per_row as usize + width as usize * 4];
            output[y * width as usize * 4..(y + 1) * width as usize * 4].copy_from_slice(src);
        }
        drop(mapped);
        resources.readback.unmap();
        Ok(())
    }

    pub fn render(
        &mut self,
        rgba: &[u8],
        width: u32,
        height: u32,
        detections: &[PoseDetection],
        kpt_conf: f32,
    ) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        self.render_inner(rgba, width, height, detections, kpt_conf, &mut output)?;
        Ok(output)
    }

    pub fn render_into(
        &mut self,
        rgba: &[u8],
        width: u32,
        height: u32,
        detections: &[PoseDetection],
        kpt_conf: f32,
        output: &mut Vec<u8>,
    ) -> Result<()> {
        self.render_inner(rgba, width, height, detections, kpt_conf, output)
    }
}

fn vertices(detections: &[PoseDetection], width: f32, height: f32, threshold: f32) -> Vec<Vertex> {
    let mut vertices = Vec::with_capacity(6 + detections.len() * 300);
    quad(
        &mut vertices,
        0.0,
        0.0,
        width,
        height,
        [0.0, 0.0, 0.0, -1.0],
        true,
        width,
        height,
    );
    for det in detections {
        let color = [0.0, 1.0, 0.45, 0.9];
        rect(
            &mut vertices,
            det.bbox.x1,
            det.bbox.y1,
            det.bbox.x2,
            det.bbox.y2,
            2.0,
            color,
            width,
            height,
        );
        for &(a, b) in &COCO_SKELETON {
            let start = det.keypoints[a];
            let end = det.keypoints[b];
            if start.conf >= threshold && end.conf >= threshold {
                line(
                    &mut vertices,
                    start.x,
                    start.y,
                    end.x,
                    end.y,
                    3.5,
                    color,
                    width,
                    height,
                );
            }
        }
        for point in det.keypoints.iter().filter(|point| point.conf >= threshold) {
            circle(
                &mut vertices,
                point.x,
                point.y,
                5.0,
                [1.0, 0.25, 0.05, 0.95],
                width,
                height,
            );
        }
    }
    vertices
}

fn vertex(x: f32, y: f32, color: [f32; 4], width: f32, height: f32) -> Vertex {
    Vertex {
        position: [x / width * 2.0 - 1.0, 1.0 - y / height * 2.0],
        uv: [x / width, y / height],
        color,
    }
}
fn quad(
    vertices: &mut Vec<Vertex>,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    color: [f32; 4],
    uv: bool,
    width: f32,
    height: f32,
) {
    let mut a = vertex(x1, y1, color, width, height);
    let mut b = vertex(x2, y1, color, width, height);
    let mut c = vertex(x2, y2, color, width, height);
    let mut d = vertex(x1, y2, color, width, height);
    if uv {
        a.uv = [0.0, 0.0];
        b.uv = [1.0, 0.0];
        c.uv = [1.0, 1.0];
        d.uv = [0.0, 1.0];
    }
    vertices.extend_from_slice(&[a, b, c, a, c, d]);
}
fn line(
    vertices: &mut Vec<Vertex>,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    thickness: f32,
    color: [f32; 4],
    width: f32,
    height: f32,
) {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let length = (dx * dx + dy * dy).sqrt();
    if length == 0.0 {
        return;
    }
    let ox = -dy / length * thickness / 2.0;
    let oy = dx / length * thickness / 2.0;
    let a = vertex(x1 + ox, y1 + oy, color, width, height);
    let b = vertex(x2 + ox, y2 + oy, color, width, height);
    let c = vertex(x2 - ox, y2 - oy, color, width, height);
    let d = vertex(x1 - ox, y1 - oy, color, width, height);
    vertices.extend_from_slice(&[a, b, c, a, c, d]);
}
fn rect(
    vertices: &mut Vec<Vertex>,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    t: f32,
    color: [f32; 4],
    w: f32,
    h: f32,
) {
    line(vertices, x1, y1, x2, y1, t, color, w, h);
    line(vertices, x2, y1, x2, y2, t, color, w, h);
    line(vertices, x2, y2, x1, y2, t, color, w, h);
    line(vertices, x1, y2, x1, y1, t, color, w, h);
}
fn circle(
    vertices: &mut Vec<Vertex>,
    x: f32,
    y: f32,
    radius: f32,
    color: [f32; 4],
    w: f32,
    h: f32,
) {
    const SEGMENTS: usize = 16;
    let center = vertex(x, y, color, w, h);
    for i in 0..SEGMENTS {
        let a = i as f32 * std::f32::consts::TAU / SEGMENTS as f32;
        let b = (i + 1) as f32 * std::f32::consts::TAU / SEGMENTS as f32;
        vertices.extend_from_slice(&[
            center,
            vertex(x + radius * a.cos(), y + radius * a.sin(), color, w, h),
            vertex(x + radius * b.cos(), y + radius * b.sin(), color, w, h),
        ]);
    }
}
