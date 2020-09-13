/// <company>Microsoft Corporation</company>
/// <copyright>Copyright (c) Microsoft Corporation 2005</copyright>
/// <summary>
///     Parity.fx 
///
///		Techniques and shaders for effects in movie maker.  Globals are set 
///		from application code using semantics.  Multiple pass techniques rely on output
///		of previous pass, unless otherwise mentioned.
/// </summary>

#include "common.fxh"

float4		g_f4FadeColor			: FadeColor;			//Color to fade to or fade from
float		g_fPosterizeLevels		: PosterizeLevels;		//Posterization levels
float		g_fPercentBrightness		: PercentBrightness;		//How much to increase brightness
float		g_fPixelationSize		: PixelationSize;		//pixel size after pixelization

//Film age globals
float		g_fNoiseFrequency		: NoiseFrequency;
float		g_fSplotchFrequency		: SplotchFrequency;
float		g_fNoiseLineFrequency		: NoiseLineFrequency;
float		g_fLintFrequency		: LintFrequency;
int		g_nBlurSize			: BlurSize;
bool		g_bGrayConvert			: GrayscaleOn;
bool 		g_bFadedEdges			: FadedEdgesOn;
float		g_fShakeFactor			: ShakeFactor;

//For Pan and zoom 
float4		g_f4TextureViewPort		: TextureViewport;

//Ripple
float2 g_f2WaveCenter	: WaveCenter;    //Origin
float2 g_f2WaveControls : WaveControls;	 //Scale to apply ripple
float g_fFrequencyConst : Frequency;     //Frequency of wave
float g_fSpeedConst		: Speed;		 //Speed
float g_fHeightConst	: Height;        //Height of wave


/************************************************************************************************/

/// <summary>
///      PS_Fade
///      
///      Fades to a color using the alpha value as the interpolator
/// </summary>
/// <param name="f2TexCoord"></param>
/// <return>
/// <para>Returns float4 type</para>
/// </return>
float4 PS_Fade(float2 f2TexCoord  : TEXCOORD0) : COLOR
{
	float4 f4TexColor = tex2D(PointSampler, f2TexCoord);
	
	float4 f4OutColor = lerp(f4TexColor, g_f4FadeColor, g_f4FadeColor.a);
	f4OutColor.a = f4TexColor.a;
	
	return f4OutColor;
}

/// <summary>
///      Fade
///      
///      Technique to fade to or from a color
/// </summary>
technique Fade
{
    pass P0
    {          
        VertexShader = compile vs_2_0 VS_Basic();
        PixelShader  = compile ps_2_0 PS_Fade(); 
    }
}


/************************************************************************************************/

/// <summary>
///      PS_Brightness
///      
///      Change brightness using a percentage of a pixel value
/// </summary>
/// <param name="f2TexCoord"></param>
/// <return>
/// <para>Returns float4 type</para>
/// </return>
float4 PS_Brightness(float2 f2TexCoord  : TEXCOORD0) : COLOR
{
	// pixel is modulated by percentage
	return tex2D(PointSampler, f2TexCoord)*(g_fPercentBrightness + 1.0f);
}

/// <summary>
///      Brightness
///      
///      Technique to modify brightness
/// </summary>
technique Brightness
{
    pass P0
    {          
        VertexShader = compile vs_2_0 VS_Basic();
        PixelShader  = compile ps_2_0 PS_Brightness(); 
    }
}


/************************************************************************************************/

/// <summary>
///      VS_PanZoom
///      
///      Vertex shader using world and texture coordinate transforms but converts a rectangle to a matrix
/// </summary>
/// <param name="f4Position">vertex position</param>
/// <param name="f2TexCoord">texture coordinate</param>
/// <return>
/// <para>Returns VS_OUTPUT for pixel shader</para>
/// </return>
VS_OUTPUT VS_PanZoom(in float4 f4Position : POSITION, in float2 f2TexCoord : TEXCOORD0 )
{

	VS_OUTPUT Output;
	
	Output.f4Position = mul(f4Position, mul(mul(mWorld,mView),mProjection));

	float4 vp = g_f4TextureViewPort;

	float3x3 f3x3Tex = {
		vp.z,  0.0, 0.0,		
		 0.0, vp.w, 0.0,		
		vp.x, vp.y, 1.0};
		
	Output.f2TexCoord = mul(float3(f2TexCoord, 1.0), f3x3Tex);
   
	return Output;
}

/// <summary>
///      PanZoom
///      
///      Technique used to pan and zoom around an image.  
/// </summary>
technique PanZoom
{
    pass P0
    {          
		//Pan and zoom around image using texture coordinate transforms
        VertexShader = compile vs_2_0 VS_Basic();
        
        //Use simple pass through shader
        PixelShader  = compile ps_2_0 PS_PanZoom(); 
    }
}


/************************************************************************************************/

/// <summary>
///      Rotate
///      
///      Technique to rotate an image
/// </summary>
technique Rotate
{
    pass P0
    {   
		//Rotation is entirely driven through vertex coordinate transforms  
		// set by the application (world matrix)
        VertexShader = compile vs_2_0 VS_Basic();
        PixelShader  = compile ps_2_0 PS_Basic(); 
    }
}

/************************************************************************************************/


/// <summary>
///      PS_Pixelate
///      
///      Decrease the resolution of the texture over time.
/// </summary>
/// <param name="f2TexCoord"></param>
/// <return>
/// <para>Returns float4 type</para>
/// </return>

float4 PS_Pixelate(float2 f2TexCoord  : TEXCOORD0) : COLOR
{
	float2 f2szI = 1.0/f2szPix;
	//float2 f2szI = f2ImageDims;
	int2 n2Dims = ceil(f2szI / ceil(g_fPixelationSize));

	float2 f2szO = (n2Dims / 2) * 2 + 1;
	
	float2 f2PixIn = f2TexCoord * f2szI;

	float2 f2SizeRatio = f2szO/f2szI;

	float2 f2Border = (f2szI-f2szO)*0.5 / f2szI;

	float2 f2OutPixel = floor(f2PixIn*f2SizeRatio + f2Border) / f2szO;
	
	return tex2D(PointSampler, f2OutPixel);
}

/// <summary>
///      Pixelate
///      
///      Technique for pixelating an image
/// </summary>
technique Pixelate
{
    pass P0
    {          
        VertexShader = compile vs_2_0 VS_Basic();
        PixelShader  = compile ps_2_0 PS_Pixelate(); 
    }
}
/************************************************************************************************/

/// <summary>
///      PS_Posterize
///      
///      Requantizes the image to a reduced color space.  
///		 For example converting 255 levels per channel to 3 levels.  The decimate space still spans
///		 the original range of colors so in the example a channel can take on (0, 127, 255)
/// </summary>
/// <param name="f2TexCoord"></param>
/// <return>
/// <para>Returns float4 type</para>
/// </return>
float4 PS_Posterize(float2 f2TexCoord  : TEXCOORD0) : COLOR
{	
	//sample, and multiply by levels and round to nearest integer which.  Dividing by number of levels
	// snaps the color to the appropriate level in the original range.
	return round(tex2D(PointSampler, f2TexCoord) * g_fPosterizeLevels)/g_fPosterizeLevels;
}

/// <summary>
///      Posterize
///      
///      Technique for posterize
/// </summary>
technique Posterize
{
    pass P0
    {          
        VertexShader = compile vs_2_0 VS_Basic();
        PixelShader  = compile ps_2_0 PS_Posterize(); 
    }
}

/************************************************************************************************/

/// <summary>
///      Grayscale
///      
///      Technique to convert to grayscale
/// </summary>
technique Grayscale
{
    pass P0
    {          
        VertexShader = compile vs_2_0 VS_Basic();
        PixelShader  = compile ps_2_0 PS_Grayscale(); 
    }
}
/************************************************************************************************/
/// <summary>
///      Sepia
///      
///      Technique to convert to sepia tone
/// </summary>
technique Sepia
{
    pass P0
    {          
        VertexShader = compile vs_2_0 VS_Basic();
        PixelShader  = compile ps_2_0 PS_Sepia(); 
    }
}
/************************************************************************************************/


/// <summary>
///      VS_KernelSepX
///      
///      Grabs a row of pixels
/// </summary>
/// <param name="f4Position"></param>
/// <param name="f4TexCoord"></param>
/// <return>
/// <para>Returns VS_OUTPUT_SEP_KERNEL type</para>
/// </return>
VS_OUTPUT_SEP_KERNEL VS_KernelSepX(in float4 f4Position : POSITION, in float2 f4TexCoord : TEXCOORD0 )
{
    return VS_KernelSep(f4Position, f4TexCoord, float2(f2szPix.x,0.0), g_nBlurSize);
}

/// <summary>
///      VS_KernelSepY
///      
///      Grabs a column of pixels
/// </summary>
/// <param name="f4Position"></param>
/// <param name="f4TexCoord"></param>
/// <return>
/// <para>Returns VS_OUTPUT_SEP_KERNEL type</para>
/// </return>
VS_OUTPUT_SEP_KERNEL VS_KernelSepY(in float4 f4Position : POSITION, in float2 f4TexCoord : TEXCOORD0 )
{
    return VS_KernelSep(f4Position, f4TexCoord, float2(0.0, f2szPix.y), g_nBlurSize);
}

float4 PS_Blur(float2 rgf2Tex[DEF_KERNELSIZE]  : TEXCOORD0) : COLOR
{
    return PS_BlurSize(rgf2Tex, g_nBlurSize);
}

/// <summary>
///      Blur
///      
///      Technique to blur using a separable kernel
/// </summary>
technique Blur
{
	//Horizontal pass
    pass P0
    {          
        VertexShader = compile vs_2_0 VS_KernelSepX();
        PixelShader  = compile ps_2_0 PS_Blur(); 
    }
    //Vertical pass 
    pass P1
    {          
        VertexShader = compile vs_2_0 VS_KernelSepY();
        PixelShader  = compile ps_2_0 PS_Blur(); 
    }

}

/************************************************************************************************/

/// <summary>
///      EdgeDetect
///      
///      Technique to perform sobel edge detection
/// </summary>
technique EdgeDetect
{
    pass P0
    {          
        VertexShader = compile vs_2_0 VS_3x3Kernel();
        PixelShader  = compile ps_2_0 PS_Sobel(); 
    }
}

/************************************************************************************************/

/// <summary>
///      PS_KuwaharaStats
///      
///      Calculates the statistics used for kuwhara edge preserved smoothing.  
/// </summary>
/// <param name="rgf2Tex">array of texture coordinates surronding the current pixel</param>
/// <return>
/// <para>Returns float4 type</para>
/// </return>
float4 PS_KuwaharaStats(float2 rgf2Tex[8]  : TEXCOORD0) : COLOR
{
	float4 rg9_f4Hood[9];
	float4 f4Out;
	
	//get colors in neighborhood.
	rg9_f4Hood[0] = tex2D(PointSampler, rgf2Tex[0]);
	rg9_f4Hood[1] = tex2D(PointSampler, rgf2Tex[1]);
	rg9_f4Hood[2] = tex2D(PointSampler, rgf2Tex[2]);
	rg9_f4Hood[3] = tex2D(PointSampler, rgf2Tex[3]);
	rg9_f4Hood[4] = tex2D(PointSampler, float2(rgf2Tex[3].x + f2szPix.x, rgf2Tex[3].y));
	rg9_f4Hood[5] = tex2D(PointSampler, rgf2Tex[4]);
	rg9_f4Hood[6] = tex2D(PointSampler, rgf2Tex[5]);
	rg9_f4Hood[7] = tex2D(PointSampler, rgf2Tex[6]);
	rg9_f4Hood[8] = tex2D(PointSampler, rgf2Tex[7]);

	// Find the per channel mean for the neighborhood
	// 
	//  --------------
	// | 1/9| 1/9| 1/9|
	// |--------------|
	// | 1/9| 1/9| 1/9|
	// |--------------|
	// | 1/9| 1/9| 1/9|
	//  --------------

	float3 f3Mean = 0.0f; 
	for(int i = 0; i < 9; i++)
	{
		f3Mean += rg9_f4Hood[i].rgb;
	}  
	f3Mean /= 9.0f;
	
	//get the variance squared as well (using variance squared is fine since we will
	// just be using these values for comparison)
	// variance squared is the sum of the squares of the diffrences betweeen each element and the mean
	// of a set.  Basically, it is a measure of how different a bunch of numbers are.  This is used later use
	// in order to find the "flattest" neighborhood.
	
	float fVarSquared = 0.0f; 
	float3 f3Diff;
	for(int n = 0; n < 9; n++)
	{
		//how different is the pixel from the average?
		f3Diff =  rg9_f4Hood[n].rgb - f3Mean;
		// square this difference and accumulate
		fVarSquared += dot(f3Diff,f3Diff);
	}  

	//pack the mean in the rgb, and the variance squared into the alpha
	f4Out.rgb = f3Mean;
	f4Out.a = fVarSquared;

	return f4Out;
}

/// <summary>
///      VS_OUTPUT_KUWAHARA_SELECT
///      
///      Stores the coordinates of for the center pixels of the 4 overlapping
///		 neighborhoods used in Kuwahara smoothing
/// </summary>
struct VS_OUTPUT_KUWAHARA_SELECT
{
    float4 f4Position   : POSITION;   // vertex position 
    float2 rgf2Tex[4]  : TEXCOORD0;	// neighboorhood centers
};

/// <summary>
///      VS_KuwaharaSelect
///      
///      Gets the center texture coordinates of the four 3x3 neighborhoods used in Kuwahara
/// </summary>
/// <param name="POSITION"></param>
/// <param name="vTexCoord"></param>
/// <return>
/// <para>Returns VS_OUTPUT_KUWAHARA_SELECT type</para>
/// </return>
VS_OUTPUT_KUWAHARA_SELECT VS_KuwaharaSelect(
				in float4 f4Position : POSITION,
				in float2 vTexCoord : TEXCOORD0 )
{
    VS_OUTPUT_KUWAHARA_SELECT Output;
	
	Output.f4Position = mul(f4Position, mul(mul(mWorld,mView),mProjection));


	float fLeft = vTexCoord.x - f2szPix.x;
	float fRight = vTexCoord.x + f2szPix.x;
	float fTop = vTexCoord.y - f2szPix.y;
	float fBottom = vTexCoord.y + f2szPix.y;
	
	
    Output.rgf2Tex[0] = float2(fLeft,		fTop);
    Output.rgf2Tex[1] = float2(fRight,		fTop);
    Output.rgf2Tex[2] = float2(fLeft,		fBottom);
    Output.rgf2Tex[3] = float2(fRight,		fBottom);
  
    return Output;
}


/// <summary>
///      PS_KuwaharaSelect
///      
///      Selects a color from the least neighborhood that is varying in color the least. 
/// </summary>
/// <param name="rgf2Tex"></param>
/// <return>
/// <para>Returns float4 type</para>
/// </return>
float4 PS_KuwaharaSelect(float2 rgf2Tex[4]  : TEXCOORD0) : COLOR
{
	//grab mean and variance squared of each 3x3 neighborhood 
	float4 rg4_f4Regions0 = tex2D(PointSampler, rgf2Tex[0]);
	float4 rg4_f4Regions1 = tex2D(PointSampler, rgf2Tex[1]);
	float4 rg4_f4Regions2 = tex2D(PointSampler, rgf2Tex[2]);
	float4 rg4_f4Regions3 = tex2D(PointSampler, rgf2Tex[3]);
	
	
	// Find the neighborhood with the least variance squared
	// and use its mean
	float4 f4Best = 0.0f;
	float4 f4SecondBest = 0.0f;
	
	float fFirstA, fSecondA;
	
	if(rg4_f4Regions0.a < rg4_f4Regions1.a)
	{
		fFirstA = rg4_f4Regions0.a;
		f4Best = rg4_f4Regions0;
	}
	else
	{
		fFirstA = rg4_f4Regions1.a;
		f4Best = rg4_f4Regions1;
	}
	
	if(rg4_f4Regions2.a < rg4_f4Regions3.a)
	{
		fSecondA = rg4_f4Regions2.a;
		f4SecondBest = rg4_f4Regions2;
	}
	else
	{
		fSecondA = rg4_f4Regions3.a;
		f4SecondBest = rg4_f4Regions3;
	}
	
	if(fSecondA < fFirstA)
	{
		f4Best = f4SecondBest;
	}

	
	return f4Best;
}

float4 PS_Sobel_Kuwahara(
    float2 rgf2Tex[8]  : TEXCOORD0) : COLOR
{
/*	float4 f4_00 = tex2D(PointSampler, rgf2Tex[0]);
	float4 f4_01 = tex2D(PointSampler, rgf2Tex[1]);
	float4 f4_02 = tex2D(PointSampler, rgf2Tex[2]);
	float4 f4_10 = tex2D(PointSampler, rgf2Tex[3]);
	float4 f4_12 = tex2D(PointSampler, rgf2Tex[4]);
	float4 f4_20 = tex2D(PointSampler, rgf2Tex[5]);
	float4 f4_21 = tex2D(PointSampler, rgf2Tex[6]);
	float4 f4_22 = tex2D(PointSampler, rgf2Tex[7]);

	float4 f4dU =	-f4_00 + 
					-2*f4_10 +
					-f4_20 +
					 f4_02 + 
					 2*f4_12 +
					 f4_22;
	
	float4 f4dV =	-f4_00 + 
					-2*f4_01 +
					-f4_02 +
					 f4_20 + 
					 2*f4_21 +
					 f4_22;
	
	float4 f4Out = 0.25*(abs(f4dU) + abs(f4dV));
*/
	float4 f4Out = 1.0 - PS_Sobel(rgf2Tex);
	
	float len = length(f4Out);
	
	float4 f4InBright = tex2D(PointSampler, float2(rgf2Tex[3].x + f2szPix.x, rgf2Tex[3].y))*1.25;
	if(len < 0.07)
	{
		return f4InBright;
	}
	else if(len < 0.15)
	{
		return f4InBright*(1.0-len);
	}
	else 
	{
		return f4InBright*(1.0-len*0.9);
	}
}

/// <summary>
///      Kuwahara
///      
///      Technique for Kuwahara smoothing and adding back edges to give a watercolor styled effect
/// </summary>
technique Kuwahara
{
	//Kuwahara is rather involved, so outside reference is appropriate for proper understanding.
	//  (It is well documented.)  Concisely, it is an edge preserving smooth filter that uses a 
	//  5x5 kernel.  This kernel is partitioned up into four overlapping 3x3 kernels from which 
	//  statistics are gathered, and used to select a mean color from the 3x3 kernel that is the
	//  most consistent in color value.  (The color changes the least over the area.)
	
    /// <summary>
    ///		Gather the mean and the variance squared for all the possible 3x3 neighborhoods in the image
    /// </summary>
    pass P0
    {
		//Make sure lighting isn't running          
		Lighting = FALSE;
		//Ensure alpha blending is off
		AlphaBlendEnable = FALSE;
        
        VertexShader = compile vs_2_0 VS_3x3Kernel();
        PixelShader  = compile ps_2_0 PS_KuwaharaStats(); 
    }
    /// <summary>
    ///      5x5 Kernel using the 3x3 neighborhoods of the previous pass
    /// </summary>
    pass P1
    {          
        VertexShader = compile vs_2_0 VS_KuwaharaSelect();
        PixelShader  = compile ps_2_0 PS_KuwaharaSelect(); 
    }
     
    /// <summary>
    ///      Bring edges back in using thresholding 
    /// </summary>
    pass PEdges
    {         
        VertexShader = compile vs_2_0 VS_3x3Kernel();
        PixelShader  = compile ps_2_0 PS_Sobel_Kuwahara(); 
    }
}


/************************************************************************************************/

struct VS_OUTPUT_POP
{
    float4 f4Position   : POSITION;   // vertex position 
    float2 f2TexCoord	: TEXCOORD0;
};

VS_OUTPUT_POP VS_Pop(in float4 f4Position : POSITION, in float2 vTexCoord : TEXCOORD0 )
{
    VS_OUTPUT_POP Output;
	
	Output.f4Position = mul(f4Position, mul(mul(mWorld,mView),mProjection));

	Output.f2TexCoord = vTexCoord;

    return Output;
}

#define WIDTH 0.01f

float4 PS_Pop(
    float2 f2Tex  : TEXCOORD0) : COLOR
{
	float2 f2BlockStart = floor(f2Tex/WIDTH)*WIDTH;
	
	float2 f2BlockCenter = f2BlockStart + WIDTH/2.0f;

	float4 f4Sample = tex2D(PointSampler, f2BlockCenter);
	
	float fBrightness = length(f4Sample.rgb)/3.0f;

	float4 f4In = tex2D(PointSampler, f2Tex);
	
	
	float fMaxChannel = max(max(f4In.r,f4In.g),f4In.b);
	
	float4 f4Modifier = 1.0f;
	float fModVal = 1.5f;
	
	if(fMaxChannel == f4In.r)
	{
		f4Modifier.r = fModVal;
	}
	else if(fMaxChannel == f4In.g)
	{
		f4Modifier.g = fModVal;
	}
	else
	{
		f4Modifier.b = fModVal;
	}
	
	if(fBrightness > 0.25)
	{
		float fDist = abs(distance(f2Tex, f2BlockCenter) / WIDTH); ///distance(f2BlockCenter, f2BlockStart));
		
		if(fDist*1.3f < fBrightness)
			return f4In*1.3;
		else
			return f4In/1.3f;
	}
	else
	{
		return f4In;
	}
}

float4 PS_Sobel_Pop(
    float2 rgf2Tex[8]  : TEXCOORD0,
    uniform sampler sampIn,
    uniform sampler sampMix) : COLOR
{
	float4 f4_00 = tex2D(sampIn, rgf2Tex[0]);
	float4 f4_01 = tex2D(sampIn, rgf2Tex[1]);
	float4 f4_02 = tex2D(sampIn, rgf2Tex[2]);
	float4 f4_10 = tex2D(sampIn, rgf2Tex[3]);
	float4 f4_12 = tex2D(sampIn, rgf2Tex[4]);
	float4 f4_20 = tex2D(sampIn, rgf2Tex[5]);
	float4 f4_21 = tex2D(sampIn, rgf2Tex[6]);
	float4 f4_22 = tex2D(sampIn, rgf2Tex[7]);

	float4 f4dU =	-f4_00 + 
					-2*f4_10 +
					-f4_20 +
					 f4_02 + 
					 2*f4_12 +
					 f4_22;
	
	float4 f4dV =	-f4_00 + 
					-2*f4_01 +
					-f4_02 +
					 f4_20 + 
					 2*f4_21 +
					 f4_22;
	
	float4 f4Out = 0.25*(abs(f4dU) + abs(f4dV));
	
	float len = length(f4Out);
	
	float4 f4EdgeColor;

	if(len < 0.10)
	{
		return tex2D(sampMix, float2(rgf2Tex[3].x + f2szPix.x, rgf2Tex[3].y));
	}
	else 
	{
		return 0.0f;
	}
}

technique PopArt
{

/*
    pass K0
    {          
		Lighting = FALSE;
		AlphaBlendEnable = FALSE;

        VertexShader = compile vs_2_0 VS_3x3Kernel();
        PixelShader  = compile ps_2_0 PS_KuwaharaStats(); 
    }
    pass K1
    {          
        VertexShader = compile vs_2_0 VS_KuwaharaSelect();
        PixelShader  = compile ps_2_0 PS_KuwaharaSelect(); 
    }
*/
    pass Poster
    {          
        VertexShader = compile vs_2_0 VS_Basic();
        PixelShader  = compile ps_2_0 PS_Posterize(); 
    }
         
    pass Pop0
    {          
		Lighting = FALSE;
		AlphaBlendEnable = FALSE;
        
        VertexShader = compile vs_2_0 VS_Pop();
        PixelShader  = compile ps_2_0 PS_Pop(); 
    }

    pass PEdges
    {
		AlphaBlendEnable = FALSE;
    	      
        VertexShader = compile vs_2_0 VS_3x3Kernel();
        PixelShader  = compile ps_2_0 PS_Sobel_Pop(LastPointSampler, PointSampler); 
    }

}


/************************************************************************************************/
#define PI 3.1415926535897932384626433832795
#define TWOPI 6.283185307179586476925286766559

#define RAD_60 1.0471975511965977461542144610932
#define RAD_120 2.0943951023931954923084289221863
#define RAD_240 4.1887902047863909846168578443727

/// <summary>
///      PS_ConvertHSI
///      
///      Convert to hue, saturation, and intensity color space. (Same as HSL, hue saturation
///			and luminance
/// </summary>
/// <param name="TEXCOORD0"></param>
/// <return>
/// <para>Returns float4 type</para>
/// </return>
float4 PS_ConvertHSI(float2 f2Tex  : TEXCOORD0) : COLOR
{
	float4 f4In = tex2D(PointSampler, f2Tex);
	
	//Calculate Hue
	float fDiff_RG = f4In.r - f4In.g;
	float fDiff_RB = f4In.r - f4In.b;
	float fDiff_GB = f4In.g - f4In.b;
	
	//Hue Calculation (angle on color wheel)
	float fHue = acos( (0.5f*(fDiff_RG + fDiff_RB)) /sqrt(pow(fDiff_RG,2) + fDiff_RB*fDiff_GB) );
	
	if(f4In.b > f4In.g)
		fHue = TWOPI - fHue;
	
	float fColSum		= dot(f4In.rgb,float3(1.0,1.0,1.0));
	float fSaturation	= 1.0 - 3.0f*min(min(f4In.r, f4In.g), f4In.b)/fColSum;
	float fIntensity	= fColSum/3.0;

	
	return float4(fHue/TWOPI, fSaturation, fIntensity, 1.0);

}


/// <summary>
///      PS_HueCycle
///      
///      
/// </summary>
/// <param name="f2TexCoord"></param>
/// <return>
/// <para>Returns float4 type</para>
/// </return>
float4 PS_HueCycle(float2 f2TexCoord  : TEXCOORD0) : COLOR
{
	//R - Hue
	//G - Saturation
	//B - Intensity
	
	float4 f4In = tex2D(PointSampler, f2TexCoord);

	float fH = f4In.r;
	float fS = f4In.g;
	float fI = f4In.b;
	
	//cycle around based on time.
	fH = fTime;
	
	fH*=TWOPI;

	int nCase = 0;
	
	if(fH > RAD_240)
	{
		nCase = 240;
		fH -= RAD_240;
	}	
	else if(fH > RAD_120)
	{
		nCase = 120;
		fH -= RAD_120;	
	}
		
	float fSatIntense = fI*(1.0-fS);
	float fHueSat =  fI*(1.0 + fS*cos(fH)/cos(RAD_60 - fH));
	float fOther = 3.0*fI - (fHueSat + fSatIntense);

	if(nCase == 0)
	{
		return float4(fHueSat, fOther, fSatIntense,1.0f);
	}
	else if(nCase == 120)
	{
		return float4(fSatIntense, fHueSat, fOther, 1.0f);
	}
	else
	{
		return float4(fOther, fSatIntense, fHueSat, 1.0f);
	}
}

/// <summary>
///		Technique for cycling through the hues
/// </summary>
technique HueCycle
{
    pass HSIConvert
    {          
        VertexShader = compile vs_2_0 VS_Basic();
        PixelShader  = compile ps_2_0 PS_ConvertHSI(); 
    }

    pass Cycle
    {          
        VertexShader = compile vs_2_0 VS_Basic();
        PixelShader  = compile ps_2_0 PS_HueCycle(); 
    }
  
}


/************************************************************************************************/

#define R_MAX_INT 2147483647
#define R_INC 27
#define R_MULT 16807


/// <summary>
///      PS_AgeColorOps
///      
///      Convert to grayscale and posterize based on parameters compile time uniform parameters
/// </summary>
/// <param name="f2TexCoord"></param>
/// <param name="bGrayConvert"></param>
/// <param name="bPosterize"></param>
/// <return>
/// <para>Returns float4 type</para>
/// </return>
float4 PS_AgeColorOps(float2 rgf2TexCoord[2]  : TEXCOORD0, uniform bool bGrayConvert, uniform bool bPosterize) : COLOR
{
	float4 f4Out;
	
	if(bGrayConvert)
	{
		f4Out = PS_Grayscale(rgf2TexCoord[0]);
	}
	else
	{
		f4Out = tex2D(PointSampler, rgf2TexCoord[0]);
	}
	
	if(bPosterize)
	{
		f4Out = round(f4Out * g_fPosterizeLevels)/g_fPosterizeLevels;
	}
	
	float4 f4NoiseColor = tex2D(NoiseLinearSampler, frac(rgf2TexCoord[1]));
	
	return  f4Out + g_fNoiseFrequency*f4NoiseColor.r;
}

float4 PS_FadedEdges(float2 f2Tex  : TEXCOORD0, uniform bool bFadedEdges) : COLOR
{
	float4 f4Out = tex2D(PointSampler, f2Tex);
	
	if(bFadedEdges)
	{
		float2 f2Center = {0.5,0.5};
		float2 f2DistCenter = f2Tex - f2Center;
		float fFade = length(f2DistCenter)/length(f2Center);
		return f4Out + fFade;
	}
	else
	{
		return f4Out;
	}
	
}

/// <summary>
///      Noise structure
/// </summary>
struct VS_OUTPUT_NOISE
{
    float4 f4Position		: POSITION;		// vertex position 
    float4 Diff			: COLOR0;		// diffuse color
};

/// <summary>
///      VS_Noise
///      
///      Simply does a pass through of the diffuse color with the standard coordinate space
///		 transform
/// </summary>
/// <param name="POSITION"></param>
/// <param name="f4Diff"></param>
/// <return>
/// <para>Returns VS_OUTPUT_NOISE type</para>
/// </return>
VS_OUTPUT_NOISE VS_Noise(in float4 f4Position : POSITION,
				in float4 f4Diff : COLOR0)
{
    VS_OUTPUT_NOISE Output;
	
	Output.f4Position = mul(f4Position, mul(mul(mWorld,mView),mProjection));

	Output.Diff = f4Diff;
       
    return Output;
}

/// <summary>      
///      Adds an aged film effect to 
/// </summary>
technique FilmAgeOld
{
    /// <summary>
    ///      Posterizes and converts to grayscale.  APPMACRO_CONVERTGRAY and APPMACRO_POSTERIZE
    ///		 are application defined preprocesser constants.  (This ensures compile time elimination
    ///		 of conditionals for the color ops.
    /// </summary>
    pass ColorOps
    {          
        VertexShader = compile vs_2_0 VS_PerturbTextureCoords();
        PixelShader  = compile ps_2_0 PS_AgeColorOps(false, false); 
    }

    /// <summary>
    ///      Blur the the image
    /// </summary>
    pass BlurX
    {          
        VertexShader = compile vs_2_0 VS_KernelSepX();
        PixelShader  = compile ps_2_0 PS_Blur(); 
    }
    pass BlurY
    {          
        VertexShader = compile vs_2_0 VS_KernelSepY();
        PixelShader  = compile ps_2_0 PS_Blur(); 
    }

    /// <summary>
    ///      Add Noise to the image using an alpha blended point list 
    /// </summary>
    pass NoiseElements
    {
        Texture[0]			= NULL;
        Texture[1]			= NULL;
        Texture[2]			= NULL;
        
        Lighting			= FALSE;
        CullMode			= NONE;
        CullMode			= NONE;
        ZEnable				= FALSE;
        
        AlphaBlendEnable		= TRUE;
        SrcBlend			= ONE; 
        DestBlend			= INVSRCALPHA; 
        
        ColorOp[0]			= SELECTARG1;
        ColorArg1[0]			= DIFFUSE;
        
        AlphaOp[0]			= SELECTARG1;
        AlphaArg1[0]			= DIFFUSE;
        
        ColorOp[1]			= DISABLE;
        AlphaOp[1]			= DISABLE;
        
        VertexShader = compile vs_2_0 VS_Noise();
        PixelShader  = NULL;
    }

    /// <summary>
    ///      Add jerkiness and faded edges around the image.  Jerkiness achieved through 
    ///		 camera (world) matrix transform.
    /// </summary>
    pass JerkinessFadedEdges
    {
		AlphaBlendEnable = FALSE;
        VertexShader = compile vs_2_0 VS_Basic();
        PixelShader  = compile ps_2_0 PS_FadedEdges(false); 
    }
}



/// <summary>      
///      Adds an aged film effect to 
/// </summary>
technique FilmAgeOlder
{
    /// <summary>
    ///      Posterizes and converts to grayscale.  APPMACRO_CONVERTGRAY and APPMACRO_POSTERIZE
    ///		 are application defined preprocesser constants.  (This ensures compile time elimination
    ///		 of conditionals for the color ops.
    /// </summary>
    pass ColorOps
    {          
        VertexShader = compile vs_2_0 VS_PerturbTextureCoords();
        PixelShader  = compile ps_2_0 PS_AgeColorOps(false, true); 
    }

    /// <summary>
    ///      Blur the the image
    /// </summary>
    pass BlurX
    {          
        VertexShader = compile vs_2_0 VS_KernelSepX();
        PixelShader  = compile ps_2_0 PS_Blur(); 
    }
    pass BlurY
    {          
        VertexShader = compile vs_2_0 VS_KernelSepY();
        PixelShader  = compile ps_2_0 PS_Blur(); 
    }

    /// <summary>
    ///      Add Noise to the image using an alpha blended point list 
    /// </summary>
    pass NoiseElements
    {
        Texture[0]			= NULL;
        Texture[1]			= NULL;
        Texture[2]			= NULL;
        
        Lighting			= FALSE;
        CullMode			= NONE;
        CullMode			= NONE;
        ZEnable				= FALSE;
        
        AlphaBlendEnable		= TRUE;
        SrcBlend			= ONE; 
        DestBlend			= INVSRCALPHA; 
        
        ColorOp[0]			= SELECTARG1;
        ColorArg1[0]			= DIFFUSE;
        
        AlphaOp[0]			= SELECTARG1;
        AlphaArg1[0]			= DIFFUSE;
        
        ColorOp[1]			= DISABLE;
        AlphaOp[1]			= DISABLE;
        
        VertexShader = compile vs_2_0 VS_Noise();
        PixelShader  = NULL;
    }

    /// <summary>
    ///      Add jerkiness and faded edges around the image.  Jerkiness achieved through 
    ///		 camera (world) matrix transform.
    /// </summary>
    pass JerkinessFadedEdges
    {
	AlphaBlendEnable = FALSE;
        VertexShader = compile vs_2_0 VS_Basic();
        PixelShader  = compile ps_2_0 PS_FadedEdges(true); 
    }
}

/// <summary>      
///      Adds an aged film effect to 
/// </summary>
technique FilmAgeOldest
{
    /// <summary>
    ///      Posterizes and converts to grayscale.  APPMACRO_CONVERTGRAY and APPMACRO_POSTERIZE
    ///		 are application defined preprocesser constants.  (This ensures compile time elimination
    ///		 of conditionals for the color ops.
    /// </summary>
    pass ColorOps
    {          
        VertexShader = compile vs_2_0 VS_PerturbTextureCoords();
        PixelShader  = compile ps_2_0 PS_AgeColorOps(true, true); 
    }

    /// <summary>
    ///      Blur the the image
    /// </summary>
    pass BlurX
    {          
        VertexShader = compile vs_2_0 VS_KernelSepX();
        PixelShader  = compile ps_2_0 PS_Blur(); 
    }
    pass BlurY
    {          
        VertexShader = compile vs_2_0 VS_KernelSepY();
        PixelShader  = compile ps_2_0 PS_Blur(); 
    }

    /// <summary>
    ///      Add Noise to the image using an alpha blended point list 
    /// </summary>
    pass NoiseElements
    {
        Texture[0]			= NULL;
        Texture[1]			= NULL;
        Texture[2]			= NULL;
        
        Lighting			= FALSE;
        CullMode			= NONE;
        CullMode			= NONE;
        ZEnable				= FALSE;
        
        AlphaBlendEnable		= TRUE;
        SrcBlend			= ONE; 
        DestBlend			= INVSRCALPHA; 
        
        ColorOp[0]			= SELECTARG1;
        ColorArg1[0]			= DIFFUSE;
        
        AlphaOp[0]			= SELECTARG1;
        AlphaArg1[0]			= DIFFUSE;
        
        ColorOp[1]			= DISABLE;
        AlphaOp[1]			= DISABLE;
        
        VertexShader = compile vs_2_0 VS_Noise();
        PixelShader  = NULL;
    }

    /// <summary>
    ///      Add jerkiness and faded edges around the image.  Jerkiness achieved through 
    ///		 camera (world) matrix transform.
    /// </summary>
    pass JerkinessFadedEdges
    {
	AlphaBlendEnable = FALSE;
        VertexShader = compile vs_2_0 VS_Basic();
        PixelShader  = compile ps_2_0 PS_FadedEdges(true); 
    }
}

/************************************************************************************************/

/// <summary>
///      PS_Laplacian
///      
///      Laplacian Kernel for sharpening
/// </summary>
/// <param name="rgf2Tex[8]">Array of texture coordinates for the pixel neighborhood</param>
/// <return>
/// <para>Returns float4 type</para>
/// </return>
float4 PS_Laplacian(float2 rgf2Tex[8]  : TEXCOORD0) : COLOR
{
	//Sample the neighborhood
	float4 f4_00 = tex2D(PointSampler, rgf2Tex[0]);
	float4 f4_01 = tex2D(PointSampler, rgf2Tex[1]);
	float4 f4_02 = tex2D(PointSampler, rgf2Tex[2]);
	float4 f4_10 = tex2D(PointSampler, rgf2Tex[3]);
	
	float4 f4_11 = tex2D(PointSampler, float2(rgf2Tex[3].x + f2szPix.x, rgf2Tex[3].y));
	
	float4 f4_12 = tex2D(PointSampler, rgf2Tex[4]);
	float4 f4_20 = tex2D(PointSampler, rgf2Tex[5]);
	float4 f4_21 = tex2D(PointSampler, rgf2Tex[6]);
	float4 f4_22 = tex2D(PointSampler, rgf2Tex[7]);

	//Calculate the Laplacian Kernel  
	//		Think of this as the second order derivative
	// 
	//  --------------
	// | -1 | -1 | -1 |
	// |--------------|
	// | -1 |  8 | -1 |
	// |--------------|
	// | -1 | -1 | -1 |
	//  --------------
	float4 f4Laplace =
					-f4_00 + 
					-f4_01 + 
					-f4_02 + 
					-f4_10 + 
					8.0 * f4_11 + 
					-f4_12 + 
					-f4_20 + 
					-f4_21 + 
					-f4_22;
	
	// Use this to amplify the pixel value.
	// Where there is lots of change, the pixel will be affected more.
	
	return f4_11 + f4Laplace;
}

/// <summary>
///      Sharpen the image.
/// </summary>
technique Sharpen
{
    pass P0
    {          
        VertexShader = compile vs_2_0 VS_3x3Kernel();
        PixelShader  = compile ps_2_0 PS_Laplacian(); 
    }
}



/************************************************************************************************/
// RIPPLE 

struct VS_RIPPLE_INPUT
{
	float4 position  : POSITION;
	float2 texCoord0 : TEXCOORD0;
};

struct VS_RIPPLE_OUTPUT 
{	
	float4 position  : POSITION;
	float2 texCoord0 : TEXCOORD0; // texture coordinates
	float3 texCoord1 : TEXCOORD1; // untransformed position 
}; 

struct PS_RIPPLE_INPUT 
{	
	float2 texCoord0 : TEXCOORD0; // texture coordinates
	float3 texCoord1 : TEXCOORD1; // untransformed position 
}; 

/// <summary>
///      Ripple vertex shader.  
/// </summary>
VS_RIPPLE_OUTPUT VS_Ripple( VS_RIPPLE_INPUT In )
{
	VS_RIPPLE_OUTPUT Out = (VS_RIPPLE_OUTPUT) 0; 

	Out.position  = mul(In.position, mul(mul(mWorld,mView),mProjection));;
	Out.texCoord0 = In.texCoord0;
	Out.texCoord1 = In.position;  //Return untransformed position for normal calcuation

	return Out;
}

/// <summary>
///      Ripple pixel shader
/// </summary>
float4 PS_Ripple( PS_RIPPLE_INPUT In ) : COLOR 
{ 
	//Find normal perturbed by ripple
	float3 f3Normal = 0; 
	
	//Determine rate and age of ripple 
	float2 f2Rates = g_f2WaveControls * 0.5 ;
	float fRippleAge = f2Rates.y * g_fSpeedConst;
	
	//Ripple radius increases over lifetime of effect
	float3 f3DistanceFromCenter = float3( (g_f2WaveCenter - In.texCoord0), 0); 
	float fRadius = fRippleAge - sqrt(dot(f3DistanceFromCenter, f3DistanceFromCenter)); 
	
	//Has ripple gotten to this pixel yet?
	if( fRadius > 0 ) 
	{ 
		float fSin = sin(-fRadius * g_fFrequencyConst); //Find sin at this pixel
		
		//Dampen height over time 
		fSin *= g_fHeightConst * f2Rates.x;
													
		f3Normal = fSin * normalize( f3DistanceFromCenter); //Move normal by sin of distance
	} 
	
	//Compute final normal
	f3Normal = f3Normal - float3(0.0, 0.0, 1.0);

	//Find vector from camera to model 
	float3 f3LookAt = 0;
	f3LookAt.xyz = normalize( mul( -In.texCoord1, mView)); 
	
	//Find new look at vector based on normal perturbed by ripple
	float3 f3NewLookAt = 0;
	f3NewLookAt.xyz = dot(f3Normal, f3LookAt) * 3 * f3Normal - (dot(f3Normal,f3Normal) * f3LookAt); 

	//Find new texture coordinate based on perturbed look at vector	
	f3NewLookAt.xy *= 1.0 / f3NewLookAt.z; 
	f3NewLookAt.xy = (-.5 * f3NewLookAt) + .5; 
	f3NewLookAt.y = 1.0 -f3NewLookAt.y;

	// sample the reflection map 
	return tex2D( PointSampler, f3NewLookAt ); 
}

/// <summary>
///      Apply ripple pixel shader
/// </summary>
technique Ripple
{
    pass P0
    {          
        VertexShader = compile vs_2_0 VS_Ripple();
        PixelShader  = compile ps_2_0 PS_Ripple();
    }
}

/******************************************************************************************/
/// <summary>
///      Add Film Noise to the image
/// </summary>
technique FilmGrain
{
    pass P0
    {          
        VertexShader = compile vs_2_0 VS_PerturbTextureCoords();
        PixelShader  = compile ps_2_0 PS_NoiseTextureOverlay();
    }
}

