from flask import Flask, render_template, request, jsonify, session, send_file
import os
import json
import base64
import requests
import traceback
import io
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import secrets
import google.generativeai as genai  # Keep old import for compatibility
# Import new SDK
try:
    from google import genai as new_genai
    from google.genai import types
    NEW_SDK_AVAILABLE = True
except ImportError:
    NEW_SDK_AVAILABLE = False
    print("New Google Gen AI SDK not available, install with: pip install google-genai")

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(16))

# Configuration
class Config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config.from_object(Config)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize OpenAI client
def get_openai_client():
    """Initialize OpenAI client with error handling"""
    if not app.config['OPENAI_API_KEY']:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    try:
        client = OpenAI(api_key=app.config['OPENAI_API_KEY'])
        return client
    except Exception as e:
        raise ValueError(f"Failed to initialize OpenAI client: {str(e)}")

# Image size configurations for social media
IMAGE_SIZES = {
    "square": {"width": 1024, "height": 1024, "label": "Square (1:1) - Instagram Post"},
    "portrait": {"width": 1024, "height": 1792, "label": "Portrait (9:16) - Instagram Story/TikTok"},
    "landscape": {"width": 1792, "height": 1024, "label": "Landscape (16:9) - YouTube/Facebook"},
    "twitter_post": {"width": 1200, "height": 675, "label": "Twitter Post (16:9)"},
    "facebook_cover": {"width": 1200, "height": 630, "label": "Facebook Cover"},
    "linkedin_post": {"width": 1200, "height": 627, "label": "LinkedIn Post"}
}

# Model configurations - Remove Gemini due to quality issues
MODELS = {
    "gpt-image-1": {
        "name": "GPT Image 1",
        "provider": "openai",
        "supports_sizes": ["square"],  # GPT Image 1 only supports 1024x1024
        "max_prompt_length": 3000
    },
    "dall-e-3": {
        "name": "DALL-E 3",
        "provider": "openai", 
        "supports_sizes": ["square", "portrait", "landscape", "twitter_post", "facebook_cover", "linkedin_post"],
        "max_prompt_length": 4000
    }
}

# Load configuration files
def load_json_config(filename, default_data):
    """Load JSON configuration with fallback to defaults"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
    return default_data

# Default configurations
DEFAULT_THEMES = {
    "Holiday Season": [
        "Christmas market scene with snow falling, warm lights, and festive decorations",
        "New Year's Eve celebration with fireworks and champagne glasses"
    ],
    "Product Showcase": [
        "Professional product photography of a luxury item on a minimalist background",
        "Lifestyle product shot in natural setting with soft lighting"
    ],
    "Corporate & Business": [
        "Modern office workspace with clean design and subtle technology elements",
        "Professional business meeting with diverse team collaborating"
    ]
}

DEFAULT_STYLES = {
    "No Style (User Prompt Only)": "",
    "70s Retro Cinematic": "Warm vintage feel with earth tones, film grain, and nostalgic lighting reminiscent of 1970s cinema",
    "Bright Studio Pop": "High-key studio lighting with vibrant colors, playful composition, and clean backgrounds"
}

# Load configurations
theme_packs = load_json_config('themes.json', DEFAULT_THEMES)
style_guides = load_json_config('style_guides.json', DEFAULT_STYLES)

def build_prompt(user_prompt, selected_style, style_influence):
    """Build a prompt that balances user input and style guide"""
    if selected_style == "No Style (User Prompt Only)" or style_influence == 0:
        return user_prompt
    
    style_guide = style_guides.get(selected_style, "")
    
    if style_influence <= 30:
        return f"{user_prompt} (Optional visual reference: {style_guide})"
    elif style_influence <= 50:
        return f"{user_prompt} Incorporate a subtle hint of this style: {style_guide}"
    elif style_influence <= 70:
        return f"Create an image showing: {user_prompt}. Use this visual style as a reference: {style_guide}"
    elif style_influence <= 90:
        return f"Using the visual aesthetics of this style: {style_guide}. Create an image that shows: {user_prompt}"
    else:
        return f"Create an image in this exact style: {style_guide}. The image should include: {user_prompt}"

def boost_prompt(original_prompt, model=None, size_key=None, selected_style=None, style_influence=None):
    """Enhanced prompt boosting that's aware of the target model, size, and visual style"""
    try:
        client = get_openai_client()
        
        # Get model and size information
        model_config = MODELS.get(model, {})
        size_config = IMAGE_SIZES.get(size_key, {})
        
        # Get style information
        style_guide = style_guides.get(selected_style, "") if selected_style else ""
        style_influence_level = int(style_influence) if style_influence else 0
        
        # Create model-specific and size-specific enhancement instructions
        enhancement_instructions = []
        
        # Base instruction
        enhancement_instructions.append(
            "You are a creative prompt enhancer for AI image generation. "
            "Your job is to take a basic image prompt and enhance it with more descriptive language, "
            "interesting details, and artistic elements while preserving the original intent."
        )
        
        # Style-specific enhancements
        if selected_style and selected_style != "No Style (User Prompt Only)" and style_influence_level > 0:
            if "blue luxe realism" in selected_style.lower():
                # Blue Luxe Realism theme (using existing style guide)
                if model == "gpt-image-1":
                    enhancement_instructions.append(
                        f"STYLE: Apply Blue Luxe Realism theme (influence: {style_influence_level}%). "
                        f"Style guide: {style_guide}. "
                        "For GPT Image 1: Focus on sharp rendering of metallics, precise color accuracy for royal blues and deep golds, "
                        "crisp white backgrounds, high-resolution editorial quality with ultra-sharp text rendering."
                    )
                else:  # DALL-E 3
                    enhancement_instructions.append(
                        f"STYLE: Apply Blue Luxe Realism theme (influence: {style_influence_level}%). "
                        f"Style guide: {style_guide}. "
                        "For DALL-E 3: Emphasize cinematic 50mm lens quality, soft daylight lighting, "
                        "photorealistic metallics, rich editorial atmosphere."
                    )
            elif "70s retro cinematic" in selected_style.lower():
                if model == "gpt-image-1":
                    enhancement_instructions.append(
                        f"STYLE: Apply 70s Retro Cinematic style (influence: {style_influence_level}%). "
                        f"Style guide: {style_guide}. "
                        "Focus on precise retro typography and sharp vintage aesthetics."
                    )
                else:  # DALL-E 3
                    enhancement_instructions.append(
                        f"STYLE: Apply 70s Retro Cinematic style (influence: {style_influence_level}%). "
                        f"Style guide: {style_guide}. "
                        "Emphasize artistic vintage cinematography and nostalgic atmosphere."
                    )
            elif "bright studio pop" in selected_style.lower():
                if model == "gpt-image-1":
                    enhancement_instructions.append(
                        f"STYLE: Apply Bright Studio Pop style (influence: {style_influence_level}%). "
                        f"Style guide: {style_guide}. "
                        "Focus on sharp, crisp rendering and precise color accuracy."
                    )
                else:  # DALL-E 3
                    enhancement_instructions.append(
                        f"STYLE: Apply Bright Studio Pop style (influence: {style_influence_level}%). "
                        f"Style guide: {style_guide}. "
                        "Emphasize vibrant pop art aesthetic and artistic studio photography feel."
                    )
            else:
                # Generic style application
                enhancement_instructions.append(
                    f"STYLE: Apply the visual style '{selected_style}' with {style_influence_level}% influence. "
                    f"Style guide: {style_guide}"
                )
        
        # Model-specific enhancements
        if model == "gpt-image-1":
            enhancement_instructions.append(
                "IMPORTANT: This is for GPT Image 1 (GPT-4o), which excels at text rendering and context awareness. "
                "Focus on adding specifications for sharp text, clear typography, professional layout, "
                "and precise visual elements. Emphasize high resolution, crisp details, and exact positioning. "
                "GPT-4o understands complex instructions, so be very specific about text placement, "
                "color schemes using hex codes if needed, and professional design elements."
            )
        elif model == "dall-e-3":
            enhancement_instructions.append(
                "IMPORTANT: This is for DALL-E 3, which excels at artistic creativity and diverse styles. "
                "Focus on artistic style descriptions, lighting, mood, composition, and creative visual elements. "
                "DALL-E 3 responds well to artistic terms like 'cinematic lighting', 'photorealistic', "
                "'hyperdetailed', and specific art styles. Emphasize creative and artistic aspects."
            )
        
        # Size-specific enhancements
        if size_config:
            aspect_ratio = size_config["width"] / size_config["height"]
            
            if aspect_ratio == 1.0:  # Square
                enhancement_instructions.append(
                    f"Size specification: This will be a SQUARE image ({size_config['width']}x{size_config['height']}). "
                    "Design for a 1:1 aspect ratio with centered composition. Ensure all elements fit within "
                    "a square format with balanced spacing. Consider circular or radial layouts."
                )
            elif aspect_ratio < 1.0:  # Portrait
                enhancement_instructions.append(
                    f"Size specification: This will be a PORTRAIT image ({size_config['width']}x{size_config['height']}). "
                    "Design for a vertical layout with top-to-bottom flow. Perfect for mobile viewing, "
                    "social media stories, or tall infographics. Stack elements vertically."
                )
            else:  # Landscape
                enhancement_instructions.append(
                    f"Size specification: This will be a LANDSCAPE image ({size_config['width']}x{size_config['height']}). "
                    "Design for a horizontal layout with left-to-right flow. Perfect for presentations, "
                    "desktop wallpapers, or wide comparison charts. Arrange elements side by side."
                )
        
        # Combine all instructions
        system_prompt = " ".join(enhancement_instructions)
        
        # Add specific prompt based on content type detection
        content_type_instructions = []
        prompt_lower = original_prompt.lower()
        
        if any(word in prompt_lower for word in ["comparison", "vs", "versus", "chart", "infographic"]):
            if model == "gpt-image-1":
                if "blue luxe realism" in selected_style.lower():
                    content_type_instructions.append(
                        "This appears to be a comparison/infographic with Blue Luxe theme. "
                        "For GPT Image 1: Create premium financial comparison chart with sophisticated blue gradient backgrounds, "
                        "gold accent borders, crystal-clear typography, professional section divisions, "
                        "editorial luxury aesthetic, and ultra-sharp text rendering."
                    )
                else:
                    content_type_instructions.append(
                        "This appears to be a comparison/infographic. For GPT Image 1: "
                        "Add specifications for clear section divisions, readable headers, "
                        "consistent typography, proper alignment, professional color scheme, "
                        "and ensure all text elements are crisp and legible."
                    )
            else:
                if "blue luxe realism" in selected_style.lower():
                    content_type_instructions.append(
                        "This appears to be a comparison/infographic with Blue Luxe theme. "
                        "For DALL-E 3: Create luxurious financial infographic with rich blue atmospheric lighting, "
                        "premium materials, sophisticated editorial elegance, and artistic depth."
                    )
                else:
                    content_type_instructions.append(
                        "This appears to be a comparison/infographic. For DALL-E 3: "
                        "Add artistic flair while maintaining readability, consider modern "
                        "design trends, clean layouts, and visually appealing color combinations."
                    )
        
        if any(word in prompt_lower for word in ["logo", "brand", "corporate"]):
            if model == "gpt-image-1":
                content_type_instructions.append(
                    "This appears to be logo/brand related. For GPT Image 1: "
                    "Focus on sharp vector-style graphics, precise geometry, "
                    "clean lines, and professional branding elements."
                )
            else:
                content_type_instructions.append(
                    "This appears to be logo/brand related. For DALL-E 3: "
                    "Add creative design elements, modern aesthetics, and artistic styling "
                    "while maintaining professional appearance."
                )
        
        # Final instruction
        final_instruction = (
            f"Enhance this image prompt for AI image generation: '{original_prompt}'\n\n"
            f"Additional context: {' '.join(content_type_instructions)}\n\n"
            "Keep the enhanced prompt under 400 words and make it detailed but focused."
        )
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_instruction}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error boosting prompt: {e}")
        return original_prompt

def add_watermark(image_bytes, logo_bytes=None):
    """Add a watermark/logo to the bottom right of an image"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        if logo_bytes is None:
            # Create text watermark
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            draw.text((img.width - 150, img.height - 30), "AI Generated", 
                     fill=(255, 255, 255, 180), font=font)
        else:
            # Add logo
            logo = Image.open(io.BytesIO(logo_bytes))
            logo_width = img.width // 8
            logo_height = int((logo.height / logo.width) * logo_width)
            logo = logo.resize((logo_width, logo_height))
            
            if logo.mode != 'RGBA':
                logo = logo.convert('RGBA')
            
            position = (img.width - logo_width - 20, img.height - logo_height - 20)
            new_img = Image.new('RGBA', img.size, (0, 0, 0, 0))
            new_img.paste(img, (0, 0))
            new_img.paste(logo, position, logo)
            img = new_img
        
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception as e:
        print(f"Error adding watermark: {e}")
        return image_bytes

def generate_with_openai(prompt, model, size_config):
    """Generate image using OpenAI models - based on working version"""
    try:
        client = get_openai_client()
        
        # Convert size config to OpenAI format
        if model == "dall-e-3":
            # DALL-E 3 supports specific size strings
            if size_config["width"] == size_config["height"]:
                size_str = "1024x1024"
            elif size_config["width"] < size_config["height"]:
                size_str = "1024x1792"
            else:
                size_str = "1792x1024"
        else:  # gpt-image-1
            size_str = "1024x1024"  # Only supported size
        
        response = client.images.generate(
            model=model,
            prompt=prompt,
            n=1,
            size=size_str
        )
        
        # Handle base64 response (gpt-image-1) - same as working version
        if hasattr(response, 'data') and response.data and hasattr(response.data[0], 'b64_json') and response.data[0].b64_json:
            image_bytes = base64.b64decode(response.data[0].b64_json)
        # Handle URL response (dall-e-3) - same as working version
        elif hasattr(response, 'data') and response.data and hasattr(response.data[0], 'url') and response.data[0].url:
            image_response = requests.get(response.data[0].url)
            if image_response.status_code == 200:
                image_bytes = image_response.content
            else:
                return None
        else:
            return None
        
        # Resize if needed (for custom social media sizes)
        if size_config["width"] != 1024 or size_config["height"] != 1024:
            img = Image.open(io.BytesIO(image_bytes))
            img = img.resize((size_config["width"], size_config["height"]), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
        
        return image_bytes
        
    except Exception as e:
        print(f"Error generating with OpenAI: {e}")
        return None

def generate_with_gemini(prompt, size_config):
    """Generate image using free Gemini 2.0 Flash model (no billing required)"""
    try:
        # Check if new SDK is available
        if not NEW_SDK_AVAILABLE:
            print("New Google Gen AI SDK not available. Install with: pip install google-genai")
            return None
            
        # Configure Gemini with new SDK
        if not app.config['GEMINI_API_KEY']:
            print("GEMINI_API_KEY not found")
            return None
        
        width, height = size_config['width'], size_config['height']
        print(f"Gemini generating image: {width}x{height} for prompt: {prompt[:50]}...")
        
        # Use the new Google Gen AI SDK with FREE Gemini model
        client = new_genai.Client(api_key=app.config['GEMINI_API_KEY'])
        
        # Generate image using FREE Gemini 2.0 Flash Preview Image Generation
        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",  # FREE model
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        
        print("Gemini API call completed")
        
        # Extract image from response
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                print(f"Gemini text response: {part.text[:100]}...")
            elif part.inline_data is not None:
                print(f"Found image data, mime_type: {part.inline_data.mime_type}")
                
                # Convert to PIL Image for resizing
                from PIL import Image
                img = Image.open(io.BytesIO(part.inline_data.data))
                print(f"Gemini generated image size: {img.size}")
                
                # Resize to exact dimensions if needed
                if img.size != (width, height):
                    img = img.resize((width, height), Image.Resampling.LANCZOS)
                    print(f"Resized to: {width}x{height}")
                
                # Convert back to bytes
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                return buffer.getvalue()
        
        print("No image found in Gemini response")
        return None
        
    except Exception as e:
        print(f"Error generating with Gemini: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_image(prompt, model="gpt-image-1", size_key="square", add_logo=False, logo_bytes=None):
    """Generate an image using the specified model and size - no fallbacks"""
    try:
        # Get model and size configurations
        if model not in MODELS:
            print(f"Unsupported model: {model}")
            return None
        
        if size_key not in IMAGE_SIZES:
            print(f"Unsupported size: {size_key}")
            return None
        
        model_config = MODELS[model]
        size_config = IMAGE_SIZES[size_key]
        
        print(f"Generating with model: {model}, size: {size_key} ({size_config['width']}x{size_config['height']})")
        
        # Generate image based on provider - NO FALLBACKS
        image_bytes = None
        
        if model_config["provider"] == "openai":
            print("Using OpenAI provider")
            image_bytes = generate_with_openai(prompt, model, size_config)
        elif model_config["provider"] == "gemini":
            print("Using Gemini provider")
            image_bytes = generate_with_gemini(prompt, size_config)
            # If Gemini fails, it fails - no fallback
        else:
            print(f"Unsupported provider: {model_config['provider']}")
            return None
        
        if not image_bytes:
            print(f"Failed to generate image with {model}")
            return None
        
        print(f"Successfully generated image with {model}: {len(image_bytes)} bytes")
        
        # Apply watermark if requested
        if add_logo:
            image_bytes = add_watermark(image_bytes, logo_bytes)
        
        return image_bytes
        
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

# Routes - based on working version
@app.route('/')
def index():
    return render_template('index.html', 
                         theme_packs=theme_packs, 
                         style_guides=style_guides)

@app.route('/boost_prompt', methods=['POST'])
def boost_prompt_route():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        model = data.get('model', 'gpt-image-1')
        size_key = data.get('size', 'square')
        selected_style = data.get('style', 'No Style (User Prompt Only)')
        style_influence = data.get('style_influence', 30)
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        boosted = boost_prompt(prompt, model, size_key, selected_style, style_influence)
        return jsonify({'boosted_prompt': boosted})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        
        user_prompt = data.get('prompt', '')
        selected_style = data.get('style', 'No Style (User Prompt Only)')
        style_influence = int(data.get('style_influence', 30))
        model = data.get('model', 'gpt-image-1')
        size_key = data.get('size', 'square')  # Now accepts size key instead of fixed size
        add_logo = data.get('add_logo', False)
        
        if not user_prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Build the full prompt - same as working version
        full_prompt = build_prompt(user_prompt, selected_style, style_influence)
        
        # Generate image with new multi-model support
        image_bytes = generate_image(
            prompt=full_prompt,
            model=model,
            size_key=size_key,
            add_logo=add_logo,
            logo_bytes=None
        )
        
        if image_bytes:
            # Convert to base64 for JSON response - same as working version
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Generate filename with model and size info
            prompt_slug = "".join(c if c.isalnum() else "_" for c in user_prompt[:30].lower())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            size_config = IMAGE_SIZES[size_key]
            filename = f"{prompt_slug}_{model}_{size_config['width']}x{size_config['height']}_{timestamp}.png"
            
            return jsonify({
                'success': True,
                'image': image_b64,
                'filename': filename,
                'prompt': full_prompt,
                'model': model,
                'size': f"{size_config['width']}x{size_config['height']}"
            })
        else:
            return jsonify({'error': 'Failed to generate image'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint for AWS"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
