// Vertex shader
struct CameraUniform {
    proj: mat4x4<f32>,
    view: mat4x4<f32>,
};

@group(1) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
	@location(0) pos: vec2<f32>,
	@location(1) uv: vec2<f32>,
}

struct InstanceInput {
	@location(2) pos: vec3<f32>,
	@location(3) color: vec3<f32>,
	@location(4) alpha: f32,
	@location(5) scale: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
	@location(0) uv: vec2<f32>,
	@location(1) alpha: f32,
};

@vertex
fn vs_main(
	model: VertexInput,
	instance: InstanceInput
) -> VertexOutput {
    var out: VertexOutput;
	out.uv = model.uv;
	out.alpha = instance.alpha;

    out.clip_position = camera.proj * camera.view
		* vec4<f32>(
			model.pos.x + instance.pos.x, 
			model.pos.y + instance.pos.y, 
			instance.pos.z + 0.0, 
			1.0
		);

    return out;
}


// Fragment shader
@group(0) @binding(0)
var hitcircle_texture: texture_2d<f32>;
@group(0) @binding(1)
var hitrcirle_texture_sampler: sampler;
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
	var hc = textureSample(hitcircle_texture, hitrcirle_texture_sampler, in.uv);
	hc.w = hc.w * in.alpha;

	return hc;
}
