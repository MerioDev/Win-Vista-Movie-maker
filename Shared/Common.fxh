// <company>Microsoft Corporation</company>
/// <copyright>Copyright (c) Microsoft Corporation 2005</copyright>
/// <summary>
///     common.fxh Commonly used shader procedures and globals for effects 
/// </summary>



//Semantic keyed globals
float4x4	mWorld				: World;			//World Matrix
float4x4	mView				: View;			//World Matrix
float4x4	mProjection			: Projection;			//World Matrix

float3x3	mTex0				: TextureMatrix0;	//Texture tranform matrix

float		fTime				: Time;				//Current time
float2		f2szPix				: PixelSize;		//Size of a pixel

texture Tex0					: InputTexture0;	//Textures
texture Tex1					: InputTexture1;
texture TexLast					: LastTexture; 
texture TexNoise				: NoiseTexture; 

//For Texture perturbation
float2		g_f2TexturePerturbation		: TexturePerturbation;

//Kernel Size
#define DEF_KERNELSIZE 			7
#define LOWER_BLUR_BOUNDARY      0.005f
#define UPPER_BLUR_BOUNDARY       0.995f

int 			g_nBlurAxis		: BlurAxis;

/// <summary>
///      Samplers (which are basically state blocks) for our textures with various filters
/// </summary>

	sampler NoiseLinearSampler = sampler_state
	{
	Texture   = (TexNoise);
	MipFilter = POINT;
	MinFilter = Anisotropic;
	MagFilter = LINEAR;
	MaxAnisotropy = 8;
	AddressU = Mirror;
	ADDRESSV = Mirror;
	ADDRESSW = Mirror;
	};

	sampler LastPointSampler = sampler_state
	{
	Texture   = (TexLast);
	MipFilter = POINT;
	MinFilter = POINT;
	MagFilter = POINT;
	AddressU = Mirror;
	ADDRESSV = Mirror;
	ADDRESSW = Mirror;
	};

	sampler LastLinearSampler = sampler_state
	{
	Texture   = (TexLast);
	MipFilter = Linear;
	MinFilter = Linear;
	MagFilter = Linear;
	AddressU = Mirror;
	ADDRESSV = Mirror;
	ADDRESSW = Mirror;
	};

	sampler PointSampler = sampler_state
	{
	Texture   = (Tex0);
	MipFilter = POINT;
	MinFilter = POINT;
	MagFilter = POINT;
	AddressU = Mirror;
	ADDRESSV = Mirror;
	ADDRESSW = Mirror;
	};

	sampler LinearSampler = sampler_state
	{
	Texture   = (Tex0);
	MipFilter = Linear;
	MinFilter = Anisotropic;
	MagFilter = LINEAR;
	MaxAnisotropy = 8;
	AddressU = Mirror;
	ADDRESSV = Mirror;
	ADDRESSW = Mirror;
	};

	struct VS_OUTPUT
	{
	float4 f4Position   : POSITION;		// vertex position
	float2 f2TexCoord	: TEXCOORD0;	// texture coordinate
	};

	/// <summary>
///      VS_Basic
///      
///      Simple vertex shader using world transform and passing the texture coordinate
///		 unmodified
/// </summary>
/// <param name="f4Position">vertex position</param>
/// <param name="f2TexCoord">texture coordinate</param>
/// <return>
/// <para>Returns VS_OUTPUT for pixel shader</para>
/// </return>
VS_OUTPUT VS_Basic(in float4 f4Position : POSITION,
				in float2 f2TexCoord : TEXCOORD0 )
{
	VS_OUTPUT Output;
	
	Output.f4Position = mul(f4Position, mul(mul(mWorld,mView),mProjection));
	Output.f2TexCoord = f2TexCoord;

	return Output;
}


/// <summary>
///      VS_BasicAndTex
///      
///      Vertex shader using world and texture coordinate transforms
/// </summary>
/// <param name="f4Position">vertex position</param>
/// <param name="f2TexCoord">texture coordinate</param>
/// <return>
/// <para>Returns VS_OUTPUT for pixel shader</para>
/// </return>
VS_OUTPUT VS_BasicAndTex(in float4 f4Position : POSITION, in float2 f2TexCoord : TEXCOORD0 )
{
    VS_OUTPUT Output;
	
	Output.f4Position = mul(f4Position, mul(mul(mWorld,mView),mProjection));
    Output.f2TexCoord = mul(float3(f2TexCoord, 1), mTex0);
   
    return Output;
}

/// <summary>
///      PS_Basic
///      
///      Samples texture 0 with a linear filter
/// </summary>
/// <param name="f4Diffuse">diffuse color</param>
/// <param name="f2TexCoord">texture coordinate</param>
/// <return>
/// <para>Returns float4 color</para>
/// </return>
float4 PS_Basic(
    float4 f4Diffuse : COLOR0,
    float2 f2TexCoord  : TEXCOORD0) : COLOR
{
	return tex2D(LinearSampler, f2TexCoord);
}


/// <summary>
///      PS_PanZoom
///      
///      The only difference between this function and PS_Basic is that this function implements a linear gradient at the edges of the quad. This helps in blurring the pixels at the edges
///      and thus implements some pseudo anti-aliasing.
/// </summary>
/// <param name="f4Diffuse">diffuse color</param>
/// <param name="f2TexCoord">texture coordinate</param>
/// <return>
/// <para>Returns float4 color</para>
/// </return>
float4 PS_PanZoom(
    float4 f4Diffuse : COLOR0,
    float2 f2TexCoord  : TEXCOORD0) : COLOR
{
	float4 f4TextureColor = tex2D(LinearSampler, f2TexCoord);
	
	return f4TextureColor*( smoothstep(0.0f , LOWER_BLUR_BOUNDARY, f2TexCoord[g_nBlurAxis]) - smoothstep(UPPER_BLUR_BOUNDARY ,  1.0f , f2TexCoord[g_nBlurAxis] ) );
}


/// <summary>
///      VS_OUTPUT_3X3KERNEL
///      
///      stores position and texture coordinates surrounding the actual current pixel
/// </summary>
struct VS_OUTPUT_3X3KERNEL
{
    float4 f4Position		: POSITION;   // vertex position 
    float2 rgf2Tex[8]		: TEXCOORD0;
};

/// <summary>
///      VS_3x3Kernel
///      
///      Vertex shader for generating surrounding texture coordinates for a pixel
///		 only 8 texture coordinates can be passed to the pixel shader, so the original
///		 texture coordinate must be generated.  Relies on the interpolator to maintain
///		 the coordinate offsets
/// </summary>
/// <param name="POSITION">vertex position</param>
/// <param name="f4TexCoord">texture coordinate</param>
/// <return>
/// <para>Returns VS_OUTPUT_3X3KERNEL type</para>
/// </return>
VS_OUTPUT_3X3KERNEL VS_3x3Kernel(in float4 f4Position : POSITION, in float2 f4TexCoord : TEXCOORD0 )
{
    VS_OUTPUT_3X3KERNEL Output;
	
	Output.f4Position = mul(f4Position, mul(mul(mWorld,mView),mProjection));


	float fLeft = f4TexCoord.x - f2szPix.x;
	float fRight = f4TexCoord.x + f2szPix.x;
	float fTop = f4TexCoord.y - f2szPix.y;
	float fBottom = f4TexCoord.y + f2szPix.y;
	
	// get the texture coordinates surrounding a texture coordinate
    Output.rgf2Tex[0] = float2(fLeft,		fTop);
    Output.rgf2Tex[1] = float2(f4TexCoord.x,	fTop);
    Output.rgf2Tex[2] = float2(fRight,		fTop);
    Output.rgf2Tex[3] = float2(fLeft,		f4TexCoord.y);
    Output.rgf2Tex[4] = float2(fRight,		f4TexCoord.y);
    Output.rgf2Tex[5] = float2(fLeft,		fBottom);
    Output.rgf2Tex[6] = float2(f4TexCoord.x,	fBottom);
    Output.rgf2Tex[7] = float2(fRight,		fBottom);
  
    return Output;
}


/// <summary>
///      VS_OUTPUT_SEP_KERNEL
///      
///      stores position and texture coordinates for a separable kernel
/// </summary>
struct VS_OUTPUT_SEP_KERNEL
{
    float4 f4Position				: POSITION;   // vertex position 
    float2 rgf2Tex[DEF_KERNELSIZE]  : TEXCOORD0;
};

/// <summary>
///      VS_KernelSep
///      
///      Generates texture coordinates the current pixel in a specified direction.  
///		 Used for separable kernels to get a column or row of pixels for the 
///		 vertical and horizontal passes. 
/// </summary>
/// <param name="f4Position">vertex position</param>
/// <param name="f4TexCoord">texture coordinate</param>
/// <param name="f2VecPix">Used to grab pixels around the center </param>
/// <return>
/// <para>Returns VS_OUTPUT_SEP_KERNEL type</para>
/// </return>
VS_OUTPUT_SEP_KERNEL VS_KernelSep(
				in float4 f4Position,
				in float2 f2TexCoord,
				in float2 f2VecPix,
				in int    nBlurSize)
{
    VS_OUTPUT_SEP_KERNEL Output;
	
	Output.f4Position = mul(f4Position, mul(mul(mWorld,mView),mProjection));

	// start from the current texture coordinate offset by the 
	// half the kernel size

	for(int i = 0; i < nBlurSize; i++)
	{
		float fOffset = i - (nBlurSize - 1)/2;
		
		Output.rgf2Tex[i] = f2TexCoord + f2VecPix*fOffset;
	}
  
    return Output;
}

/************************************************************************************************/

const float4 f4GrayConvert = {0.299f, 0.587f, 0.114f, 0.0f};

/// <summary>
///      PS_Grayscale
///      
///      
/// </summary>
/// <param name="f2TexCoord"></param>
/// <return>
/// <para>Returns float4 type</para>
/// </return>
float4 PS_Grayscale(float2 f2TexCoord  : TEXCOORD0) : COLOR
{	
	return dot(tex2D(PointSampler, f2TexCoord), f4GrayConvert);
}


//Sepia tone conversion factor
const float4 f4SepiaConvert = {1.1f, 0.91f, 0.75f, 1.0f};

/// <summary>
///      PS_Sepia
///      
///      Convert to sepia tone
/// </summary>
/// <param name="f2TexCoord"></param>
/// <return>
/// <para>Returns float4 type</para>
/// </return>
float4 PS_Sepia(
    float2 f2TexCoord  : TEXCOORD0) : COLOR
{	
	return PS_Grayscale(f2TexCoord) * f4SepiaConvert;
}


/// <summary>
///      PS_Blur
///      
///      Get average color value  
/// </summary>
/// <param name="rgf2Tex">array of texture coordinates from which to sample </param>
/// <return>
/// <para>Returns float4 color</para>
/// </return>
float4 PS_BlurSize(float2 rgf2Tex[DEF_KERNELSIZE]  : TEXCOORD0, in int nBlurSize) : COLOR
{
	float4 f4Out = {0.0f, 0.0f, 0.0f, 0.0f};
	for(int i = 0; i < nBlurSize; i++)
	{
		f4Out += tex2D(PointSampler, rgf2Tex[i]);
	}
	return f4Out/float(nBlurSize);
}


/// <summary>
///      PS_Sobel
///      
///      Performs sobel edge detection
/// </summary>
/// <param name="rgf2Tex">array of texture coordinates from which to sample </param>
/// <return>
/// <para>Returns float4 color representing the edge (white is no edge)</para>
/// </return>
float4 PS_Sobel(float2 rgf2Tex[8]  : TEXCOORD0) : COLOR
{
	//Sample neighbors	
	float4 f4_00 = tex2D(PointSampler, rgf2Tex[0]);
	float4 f4_01 = tex2D(PointSampler, rgf2Tex[1]);
	float4 f4_02 = tex2D(PointSampler, rgf2Tex[2]);
	float4 f4_10 = tex2D(PointSampler, rgf2Tex[3]);
	float4 f4_12 = tex2D(PointSampler, rgf2Tex[4]);
	float4 f4_20 = tex2D(PointSampler, rgf2Tex[5]);
	float4 f4_21 = tex2D(PointSampler, rgf2Tex[6]);
	float4 f4_22 = tex2D(PointSampler, rgf2Tex[7]);

	//Apply horizontal kernel
	// 
	//  --------------
	// | -1 |  0 |  1 |
	// |--------------|
	// | -2 |  0 |  2 |
	// |--------------|
	// | -1 |  0 |  1 |
	//  --------------

	float4 f4dU =	-f4_00 + 
					-2*f4_10 +
					-f4_20 +
					 f4_02 + 
					 2*f4_12 +
					 f4_22;
	
	//Apply vertical kernel
	// 
	//  --------------
	// | -1 | -2 | -1 |
	// |--------------|
	// |  0 |  0 |  0 |
	// |--------------|
	// |  1 |  2 |  1 |
	//  --------------

	float4 f4dV =	-f4_00 + 
					-2*f4_01 +
					-f4_02 +
					 f4_20 + 
					 2*f4_21 +
					 f4_22;
					
	return 1.0 - 0.25*(abs(f4dU) + abs(f4dV));
}

/*****************************************************/
/// <summary>
///      VS_OUTPUT_3X3KERNEL
///      
///      stores position and texture coordinates surrounding the actual current pixel
/// </summary>
struct VS_OUTPUT_TWOTEXCOORD
{
    float4 f4Position		: POSITION;   // vertex position 
    float2 rgf2Tex[2]		: TEXCOORD0;
};

/// <summary>
///      VS_PerturbTextureCoords
///      
///      Vertex shader for generating surrounding texture coordinates for a pixel
///		 only 8 texture coordinates can be passed to the pixel shader, so the original
///		 texture coordinate must be generated.  Relies on the interpolator to maintain
///		 the coordinate offsets
/// </summary>
/// <param name="POSITION">vertex position</param>
/// <param name="f4TexCoord">texture coordinate</param>
/// <return>
/// <para>Returns VS_OUTPUT_TWOTEXCOORD type</para>
/// </return>
VS_OUTPUT_TWOTEXCOORD VS_PerturbTextureCoords(
	in float4 f4Position : POSITION, 
	in float2 f4TexCoord : TEXCOORD0 )
{	
	VS_OUTPUT_TWOTEXCOORD Output;
	Output.f4Position = mul(f4Position, mul(mul(mWorld,mView),mProjection));

	Output.rgf2Tex[0] = f4TexCoord;

	//offset by global
	Output.rgf2Tex[1] = f4TexCoord + g_f2TexturePerturbation;
	
	return Output;
}


/// <summary>
///      PS_NoiseTextureOverlay
///      
///      Fades to a color using the alpha value as the interpolator
/// </summary>
/// <param name="rgf2TexCoord"></param>
/// <return>
/// <para>Returns float4 type</para>
/// </return>
float4 PS_NoiseTextureOverlay(float2 rgf2TexCoord[2]  : TEXCOORD0) : COLOR
{
	float4 f4InTexColor = tex2D(PointSampler, rgf2TexCoord[0]);
	float4 f4NoiseColor = tex2D(NoiseLinearSampler, frac(rgf2TexCoord[1]));
	
	return  f4InTexColor + 0.25*f4NoiseColor;
}
