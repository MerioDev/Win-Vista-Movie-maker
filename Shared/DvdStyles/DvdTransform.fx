// transformations
float4x4 mWorld                 : WORLD;
float4x4 mView                  : VIEW;
float4x4 mProjection            : PROJECTION;

float4 rcVideoInInput           : VideoRectInInput;
float4 rcVideoInOutput          : VideoRectInOutput;
float flVideoZoom               : VideoZoom;

// vertex shader output structure
struct VS_OUTPUT
{
    float4 Pos                  : POSITION;
    float2 Tex0                 : TEXCOORD0;
};

float flTime                    : Time;

texture VideoTex                : VideoTexture;
texture MaskTex                 : MaskTexture;
texture OverlayTex              : OverlayTexture;
texture UnderlayTex             : UnderlayTexture;

float4 OverlayColor             : OverlayColor;

float MasterAlpha               : MasterAlpha;

float2 fl2VideoCellSize         : VideoCellSize;
float2 fl2VideoGridLineWidth    : VideoGridLineWidth;

sampler VideoSampler = sampler_state
{
    Texture = (VideoTex);
    MipFilter = Linear;
    MinFilter = Linear;
    MagFilter = Linear;
};

sampler MaskSampler = sampler_state
{
    Texture = (MaskTex);
    MipFilter = Linear;
    MinFilter = Linear;
    MagFilter = Linear;
};

sampler OverlaySampler = sampler_state
{
    Texture = (OverlayTex);
    MipFilter = Linear;
    MinFilter = Linear;
    MagFilter = Linear;
};

sampler SubpictureSampler = sampler_state
{
    Texture = (OverlayTex);
    MipFilter = Point;
    MinFilter = Point;
    MagFilter = Point;
};

sampler UnderlaySampler = sampler_state
{
    Texture = (UnderlayTex);
    MipFilter = Linear;
    MinFilter = Linear;
    MagFilter = Linear;
};

float AlphaFromRGB(float4 colorIn)
{
    float OutAlpha = (colorIn.r + colorIn.g + colorIn.b) / 3.0f;

	if (OutAlpha < 0.1f)
	{
		OutAlpha = 0.0f;
	}
	else if (OutAlpha > 0.9f)
	{
		OutAlpha = 1.0f;
	}

    return OutAlpha;
}

float4 CombineColors(float4 color1, float4 color2)
{
    float OutAlpha = 1.0f - ((1.0f - color1.a) * (1.0f - color2.a));

    float4 OutColor = 0.0f;

    if (OutAlpha != 0.0f)
    {
        float alphaFactor2 = color2.a / OutAlpha;
        float alphaFactor1 = 1.0f - alphaFactor2;

        OutColor = (alphaFactor1 * color1) + (alphaFactor2 * color2);
        OutColor.a = OutAlpha;
    }

    return OutColor;
}

VS_OUTPUT VS_DVD(
    in float4 InPos : POSITION,
    in float2 tex0  : TEXCOORD0)
{
    VS_OUTPUT Out = (VS_OUTPUT)0;
    
    // transform the position
    Out.Pos = mul(mul(mul(InPos, mWorld), mView), mProjection);
    Out.Tex0 = tex0;

    return Out;
}

float4 PS_DVD_ZoomOut(
    float2 tex0  : TEXCOORD0,
    uniform bool fUseVideo,
    uniform bool fUseMask,
    uniform bool fUseUnderlayTex,
    uniform bool fUseOverlayTex,
    uniform bool fUseOverlayColor,
    uniform bool fUseAlphaFromRGB) : COLOR
{
    float4 color = 0.0f;

    if (fUseVideo)
    {
        // rcVideoInOutput is really a rect of the form
        // x = left, y = top, z = width, w = height.
        if (tex0.x >= rcVideoInOutput.x &&
            tex0.x <= (rcVideoInOutput.x + rcVideoInOutput.z) &&
            tex0.y >= rcVideoInOutput.y &&
            tex0.y <= (rcVideoInOutput.y + rcVideoInOutput.w))
        {
            float2 TexForSampler;
            TexForSampler.x = (tex0.x - rcVideoInOutput.x) / rcVideoInOutput.z;
            TexForSampler.y = (tex0.y - rcVideoInOutput.y) / rcVideoInOutput.w;
            
            float flInverseVideoZoom = 1.0f / flVideoZoom;

			color.a = 1.0f;
			
			if (flVideoZoom > 1.0f)
			{
				//float flCropped = (1.0f - (1.0f / flVideoZoom)) / 2.0f;
				float flCropped = (1.0f - flInverseVideoZoom) / 2.0f;
				if ((TexForSampler.x >= flCropped) &&
					(TexForSampler.x <= 1.0f - flCropped))
				{
					TexForSampler.x = (flCropped - TexForSampler.x) / (2.0f * flCropped - 1.0f);
					color = tex2D(VideoSampler, TexForSampler);
				}
			}
			else
			{
				float flCropped = (1.0f - flVideoZoom) / 2.0f;
				if ((TexForSampler.y >= flCropped) &&
					(TexForSampler.y <= 1.0f - flCropped))
				{
					TexForSampler.y = (flCropped - TexForSampler.y) / (2.0f * flCropped - 1.0f);
					color = tex2D(VideoSampler, TexForSampler);
				}
			}
        }
    }

    if (fUseMask)
    {
        float4 maskColor = tex2D(MaskSampler, tex0);

        if(fUseAlphaFromRGB)
        {
            color.a = AlphaFromRGB(maskColor);
        }
        else
        {
			color.a = maskColor.a;
        }
    }
    
    if (fUseUnderlayTex)
    {
		float4 underlayColor = tex2D(UnderlaySampler, tex0);
        color = CombineColors(underlayColor, color);
    }

    if (fUseOverlayTex)
    {
        color = CombineColors(color, tex2D(OverlaySampler, tex0));
    }

    if (fUseOverlayColor)
    {
        color = CombineColors(color, OverlayColor);
    }

    color.a *= MasterAlpha;

	return color;
}

float4 PS_DVD(
    float2 tex0  : TEXCOORD0,
    uniform bool fUseVideo,
    uniform bool fUseMask,
    uniform bool fUseUnderlayTex,
    uniform bool fUseOverlayTex,
    uniform bool fUseOverlayColor,
    uniform bool fUseAlphaFromRGB) : COLOR
{
    float4 color = 0.0f;

    if (fUseVideo)
    {
        // rcVideoInOutput is really a rect of the form
        // x = left, y = top, z = width, w = height.
        if (tex0.x >= rcVideoInOutput.x &&
            tex0.x <= (rcVideoInOutput.x + rcVideoInOutput.z) &&
            tex0.y >= rcVideoInOutput.y &&
            tex0.y <= (rcVideoInOutput.y + rcVideoInOutput.w))
        {
            float2 TexForSampler;
            TexForSampler.x = (tex0.x - rcVideoInOutput.x) / rcVideoInOutput.z;
            TexForSampler.y = (tex0.y - rcVideoInOutput.y) / rcVideoInOutput.w;
            
            float flInverseVideoZoom = 1.0f / flVideoZoom;

			// Maintains aspect ratio and zooms content (zoom in)
			if (flVideoZoom < 1.0f)
			{
				float flCropped = (1.0f - flVideoZoom) / 2.0f;
				TexForSampler.x = lerp(flCropped, flVideoZoom + flCropped, TexForSampler.x);
			}
			else
			{
				float flCropped = (1.0f - flInverseVideoZoom) / 2.0f;
				TexForSampler.y = lerp(flCropped, flInverseVideoZoom + flCropped, TexForSampler.y);
			}
			TexForSampler = lerp(float2(rcVideoInInput.x, rcVideoInInput.y),
				float2(rcVideoInInput.x + rcVideoInInput.z, rcVideoInInput.y + rcVideoInInput.w), TexForSampler);
				
			color = tex2D(VideoSampler, TexForSampler);
        }
    }

    if (fUseMask)
    {
        float4 maskColor = tex2D(MaskSampler, tex0);

        if(fUseAlphaFromRGB)
        {
            color.a = AlphaFromRGB(maskColor);
        }
        else
        {
			color.a = maskColor.a;
        }
    }
    
    if (fUseUnderlayTex)
    {
		float4 underlayColor = tex2D(UnderlaySampler, tex0);
        color = CombineColors(underlayColor, color);
    }

    if (fUseOverlayTex)
    {
        color = CombineColors(color, tex2D(OverlaySampler, tex0));
    }

    if (fUseOverlayColor)
    {
        color = CombineColors(color, OverlayColor);
    }

    color.a *= MasterAlpha;

    return color;
}

float4 PS_DVD_WithoutZoom(
    float2 tex0  : TEXCOORD0,
    uniform bool fUseVideo,
    uniform bool fUseMask,
    uniform bool fUseUnderlayTex,
    uniform bool fUseOverlayTex,
    uniform bool fUseOverlayColor,
    uniform bool fUseAlphaFromRGB) : COLOR
{
    float4 color = 0.0f;

    if (fUseVideo)
    {
        // rcVideoInOutput is really a rect of the form
        // x = left, y = top, z = width, w = height.
        if (tex0.x >= rcVideoInOutput.x &&
            tex0.x <= (rcVideoInOutput.x + rcVideoInOutput.z) &&
            tex0.y >= rcVideoInOutput.y &&
            tex0.y <= (rcVideoInOutput.y + rcVideoInOutput.w))
        {
            float2 TexForSampler;
            TexForSampler.x = (tex0.x - rcVideoInOutput.x) / rcVideoInOutput.z;
            TexForSampler.y = (tex0.y - rcVideoInOutput.y) / rcVideoInOutput.w;

            TexForSampler = lerp(float2(rcVideoInInput.x, rcVideoInInput.y),
                float2(rcVideoInInput.x + rcVideoInInput.z, rcVideoInInput.y + rcVideoInInput.w), TexForSampler);
            color = tex2D(VideoSampler, TexForSampler);
        }
    }

    if (fUseMask)
    {
        float4 maskColor = tex2D(MaskSampler, tex0);

        if(fUseAlphaFromRGB)
        {
            color.a = AlphaFromRGB(maskColor);
        }
        else
        {
			color.a = maskColor.a;
        }
    }
    
    if (fUseUnderlayTex)
    {
		float4 underlayColor = tex2D(UnderlaySampler, tex0);
        color = CombineColors(underlayColor, color);
    }

    if (fUseOverlayTex)
    {
        color = CombineColors(color, tex2D(OverlaySampler, tex0));
    }
    
    if (fUseOverlayColor)
    {
        color = CombineColors(color, OverlayColor);
    }

    color.a *= MasterAlpha;

    return color;
}

float4 PS_DVD_Subpicture(
    float2 tex0  : TEXCOORD0) : COLOR
{
	return tex2D(SubpictureSampler, tex0);
}

technique Video
{
    pass P0
    {
        CullMode = NONE;
        AlphaBlendEnable = TRUE;
        SrcBlend = SRCALPHA;
        DestBlend = INVSRCALPHA;

        VertexShader = compile vs_2_0 VS_DVD();
        PixelShader  = compile ps_2_0 PS_DVD(true, false, false, false, false, false);
    }
}

technique VideoZoomOut
{
    pass P0
    {
        CullMode = NONE;
        AlphaBlendEnable = TRUE;
        SrcBlend = SRCALPHA;
        DestBlend = INVSRCALPHA;

        VertexShader = compile vs_2_0 VS_DVD();
        PixelShader  = compile ps_2_0 PS_DVD_ZoomOut(true, false, false, false, false, false);
    }
}

technique VideoMask
{
    pass P0
    {
        CullMode = NONE;
        AlphaBlendEnable = TRUE;
        SrcBlend = SRCALPHA;
        DestBlend = INVSRCALPHA;

        VertexShader = compile vs_2_0 VS_DVD();
        PixelShader  = compile ps_2_0 PS_DVD(true, true, false, false, false, false);
    }
}

technique VideoMaskZoomOut
{
    pass P0
    {
        CullMode = NONE;
        AlphaBlendEnable = TRUE;
        SrcBlend = SRCALPHA;
        DestBlend = INVSRCALPHA;

        VertexShader = compile vs_2_0 VS_DVD();
        PixelShader  = compile ps_2_0 PS_DVD_ZoomOut(true, true, false, false, false, false);
    }
}

technique VideoMaskWithoutZoom
{
    pass P0
    {
        CullMode = NONE;
        AlphaBlendEnable = TRUE;
        SrcBlend = SRCALPHA;
        DestBlend = INVSRCALPHA;

        VertexShader = compile vs_2_0 VS_DVD();
        PixelShader  = compile ps_2_0 PS_DVD_WithoutZoom(true, true, false, false, false, false);
    }
}

technique VideoSphere
{
    pass P0
    {
        CullMode = CCW;
        ZEnable = TRUE;
        AlphaBlendEnable = TRUE;
        SrcBlend = SRCALPHA;
        DestBlend = INVSRCALPHA;

        VertexShader = compile vs_2_0 VS_DVD();
        PixelShader  = compile ps_2_0 PS_DVD(true, false, false, true, false, false);
    }
}

technique VideoMaskSphere
{
    pass P0
    {
        CullMode = CCW;
        ZEnable = TRUE;
        AlphaBlendEnable = TRUE;
        SrcBlend = SRCALPHA;
        DestBlend = INVSRCALPHA;

        VertexShader = compile vs_2_0 VS_DVD();
        PixelShader  = compile ps_2_0 PS_DVD(true, true, false, false, false, false);
    }
}

technique VideoRGBMask
{
    pass P0
    {
        CullMode = NONE;
        AlphaBlendEnable = TRUE;
        SrcBlend = SRCALPHA;
        DestBlend = INVSRCALPHA;

        VertexShader = compile vs_2_0 VS_DVD();
        PixelShader  = compile ps_2_0 PS_DVD_WithoutZoom(true, true, false, false, false, true);
    }
}

technique VideoRGBMaskZoomOut
{
    pass P0
    {
        CullMode = NONE;
        AlphaBlendEnable = TRUE;
        SrcBlend = SRCALPHA;
        DestBlend = INVSRCALPHA;

        VertexShader = compile vs_2_0 VS_DVD();
        PixelShader  = compile ps_2_0 PS_DVD_ZoomOut(true, true, false, false, false, true);
    }
}

technique OverlayTex
{
    pass P0
    {
        CullMode = NONE;
        AlphaBlendEnable = TRUE;
        SrcBlend = SRCALPHA;
        DestBlend = INVSRCALPHA;

        VertexShader = compile vs_2_0 VS_DVD();
        PixelShader  = compile ps_2_0 PS_DVD(false, false, false, true, false, false);
    }
}

technique SubpictureTex
{
    pass P0
    {
        CullMode = NONE;
        AlphaBlendEnable = TRUE;
        SrcBlend = SRCALPHA;
        DestBlend = INVSRCALPHA;

        VertexShader = compile vs_2_0 VS_DVD();
        PixelShader  = compile ps_2_0 PS_DVD_Subpicture();
    }
}

technique OverlayColor
{
    pass P0
    {
        CullMode = NONE;
        AlphaBlendEnable = TRUE;
        SrcBlend = SRCALPHA;
        DestBlend = INVSRCALPHA;

        VertexShader = compile vs_2_0 VS_DVD();
        PixelShader  = compile ps_2_0 PS_DVD(false, false, false, false, true, false);
    }
}

technique VideoOverlayTex
{
    pass P0
    {
        CullMode = NONE;
        AlphaBlendEnable = TRUE;
        SrcBlend = SRCALPHA;
        DestBlend = INVSRCALPHA;

        VertexShader = compile vs_2_0 VS_DVD();
        PixelShader  = compile ps_2_0 PS_DVD(true, false, false, true, false, false);
    }
}

technique VideoOverlayTexZoomOut
{
    pass P0
    {
        CullMode = NONE;
        AlphaBlendEnable = TRUE;
        SrcBlend = SRCALPHA;
        DestBlend = INVSRCALPHA;

        VertexShader = compile vs_2_0 VS_DVD();
        PixelShader  = compile ps_2_0 PS_DVD_ZoomOut(true, false, false, true, false, false);
    }
}

technique VideoOverlayColor
{
    pass P0
    {
        CullMode = NONE;
        AlphaBlendEnable = TRUE;
        SrcBlend = SRCALPHA;
        DestBlend = INVSRCALPHA;

        VertexShader = compile vs_2_0 VS_DVD();
        PixelShader  = compile ps_2_0 PS_DVD(true, false, false, false, true, false);
    }
}

technique VideoMaskOverlayTex
{
    pass P0
    {
        CullMode = NONE;
        AlphaBlendEnable = TRUE;
        SrcBlend = SRCALPHA;
        DestBlend = INVSRCALPHA;

        VertexShader = compile vs_2_0 VS_DVD();
        PixelShader  = compile ps_2_0 PS_DVD(true, true, false, true, false, false);
    }
}

technique VideoMaskOverlayTexZoomOut
{
    pass P0
    {
        CullMode = NONE;
        AlphaBlendEnable = TRUE;
        SrcBlend = SRCALPHA;
        DestBlend = INVSRCALPHA;

        VertexShader = compile vs_2_0 VS_DVD();
        PixelShader  = compile ps_2_0 PS_DVD_ZoomOut(true, true, false, true, false, false);
    }
}

technique VideoMaskUnderlayTex
{
    pass P0
    {
        CullMode = NONE;
        AlphaBlendEnable = TRUE;
        SrcBlend = SRCALPHA;
        DestBlend = INVSRCALPHA;

        VertexShader = compile vs_2_0 VS_DVD();
        PixelShader  = compile ps_2_0 PS_DVD(true, true, true, false, false, false);
    }
}

technique VideoMaskUnderlayOverlayTex
{
    pass P0
    {
        CullMode = NONE;
        AlphaBlendEnable = TRUE;
        SrcBlend = SRCALPHA;
        DestBlend = INVSRCALPHA;

        VertexShader = compile vs_2_0 VS_DVD();
        PixelShader  = compile ps_2_0 PS_DVD(true, true, true, true, false, false);
    }
}

technique VideoMaskOverlayColor
{
    pass P0
    {
        CullMode = NONE;
        AlphaBlendEnable = TRUE;
        SrcBlend = SRCALPHA;
        DestBlend = INVSRCALPHA;

        VertexShader = compile vs_2_0 VS_DVD();
        PixelShader  = compile ps_2_0 PS_DVD(true, true, false, false, true, false);
    }
}

technique VideoMaskOverlayTexOverlayColor
{
    pass P0
    {
        CullMode = NONE;
        AlphaBlendEnable = TRUE;
        SrcBlend = SRCALPHA;
        DestBlend = INVSRCALPHA;

        VertexShader = compile vs_2_0 VS_DVD();
        PixelShader  = compile ps_2_0 PS_DVD(true, true, false, true, true, false);
    }
}



float4 PS_VideoWallGrid(
    float2 tex0  : TEXCOORD0) : COLOR
{
    float4 color = 0.0f;

    float2 fl2Ip; // integer portion;
    modf(tex0 / fl2VideoCellSize, fl2Ip);

    if (((tex0.x - (fl2Ip.x * fl2VideoCellSize.x)) <= fl2VideoGridLineWidth.x) ||
        ((tex0.y - (fl2Ip.y * fl2VideoCellSize.y)) <= fl2VideoGridLineWidth.y) ||
        ((((1.0f + fl2Ip.x) * fl2VideoCellSize.x) - tex0.x) <= fl2VideoGridLineWidth.x) ||
        ((((1.0f + fl2Ip.y) * fl2VideoCellSize.y) - tex0.y) <= fl2VideoGridLineWidth.y))
    {
        color.a = 1.0f;
    }

    return color;
}

technique VideoWallGrid
{
    pass P0
    {
        CullMode = NONE;
        AlphaBlendEnable = TRUE;
        SrcBlend = SRCALPHA;
        DestBlend = INVSRCALPHA;

        VertexShader = compile vs_2_0 VS_DVD();
        PixelShader  = compile ps_2_0 PS_VideoWallGrid();
    }
}
