// Vertex shader
struct CameraUniform {
    proj: mat4x4<f32>,
    view: mat4x4<f32>,
};

@group(1) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
	@location(0) pos: vec3<f32>,
	@location(1) uv: vec2<f32>,
}

struct InstanceInput {
	@location(2) pos: vec3<f32>,
	@location(3) alpha: f32,
	@location(4) scale: f32,
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

    let clip_position = camera.proj * camera.view
		* vec4<f32>(
			(model.pos.x * instance.scale) + instance.pos.x,
			(model.pos.y * instance.scale) + instance.pos.y,
			instance.pos.z,
			1.0
		);

    var out: VertexOutput;
	out.uv = model.uv;
	out.alpha = instance.alpha;
	out.clip_position = clip_position;
    return out;
}


// Fragment shader
@group(0) @binding(0)
var texture: texture_2d<f32>;
@group(0) @binding(1)
var texture_sampler: sampler;


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
	var out = textureSample(texture, texture_sampler, in.uv);

	out.w = out.w * in.alpha;


	return out;
	//return vec4<f32>(1.0, 0.2, 0.1, in.alpha);
	//return vec4<f32>(1.0, 0.2, 0.1, in.alpha);
}
