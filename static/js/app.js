// Model and size configurations
const MODELS = {
    "gpt-image-1": {
        "name": "GPT Image 1",
        "provider": "openai",
        "supports_sizes": ["square"],
        "max_prompt_length": 1000,
        "description": "Fast and efficient model, square images only"
    },
    "dall-e-3": {
        "name": "DALL-E 3",
        "provider": "openai", 
        "supports_sizes": ["square", "portrait", "landscape", "twitter_post", "facebook_cover", "linkedin_post"],
        "max_prompt_length": 4000,
        "description": "High-quality detailed images, all sizes supported"
    }
};

const IMAGE_SIZES = {
    "square": {"width": 1024, "height": 1024, "label": "Square (1:1) - Instagram Post", "icon": "square"},
    "portrait": {"width": 1024, "height": 1792, "label": "Portrait (9:16) - Instagram Story/TikTok", "icon": "portrait"},
    "landscape": {"width": 1792, "height": 1024, "label": "Landscape (16:9) - YouTube/Facebook", "icon": "landscape"},
    "twitter_post": {"width": 1200, "height": 675, "label": "Twitter Post (16:9)", "icon": "landscape"},
    "facebook_cover": {"width": 1200, "height": 630, "label": "Facebook Cover", "icon": "landscape"},
    "linkedin_post": {"width": 1200, "height": 627, "label": "LinkedIn Post", "icon": "landscape"}
};

// Global variables for DOM elements
let elements = {};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing app...');
    
    // Get all DOM elements
    initializeElements();
    
    // Set up event listeners
    setupEventListeners();
    
    // Initialize UI state
    updateModelInfo();
    updateSizePreview();
    updateAvailableSizes();
    
    console.log('App initialized successfully');
});

function initializeElements() {
    elements = {
        promptInput: document.getElementById('prompt'),
        modelSelect: document.getElementById('model'),
        imageSizeSelect: document.getElementById('imageSize'),
        styleSelect: document.getElementById('style'),
        styleInfluence: document.getElementById('styleInfluence'),
        styleInfluenceValue: document.getElementById('styleInfluenceValue'),
        addLogoCheckbox: document.getElementById('addLogo'),
        generateBtn: document.getElementById('generateBtn'),
        boostPromptBtn: document.getElementById('boostPromptBtn'),
        advancedSettingsToggle: document.getElementById('advancedSettingsToggle'),
        advancedSettings: document.getElementById('advancedSettings'),
        previewPlaceholder: document.getElementById('previewPlaceholder'),
        previewImage: document.getElementById('previewImage'),
        generatedImage: document.getElementById('generatedImage'),
        loadingOverlay: document.getElementById('loadingOverlay'),
        previewDetails: document.getElementById('previewDetails'),
        usedPrompt: document.getElementById('usedPrompt'),
        usedModelSize: document.getElementById('usedModelSize'),
        downloadBtn: document.getElementById('downloadBtn'),
        copyPromptBtn: document.getElementById('copyPromptBtn'),
        newImageBtn: document.getElementById('newImageBtn'),
        toast: document.getElementById('toast'),
        toastMessage: document.getElementById('toastMessage'),
        modelInfo: document.getElementById('modelInfo'),
        sizePreview: document.getElementById('sizePreview')
    };
    
    console.log('Elements initialized:', Object.keys(elements).length + ' elements found');
}

function setupEventListeners() {
    console.log('Setting up event listeners...');
    
    // Model change handler
    if (elements.modelSelect) {
        elements.modelSelect.addEventListener('change', function() {
            console.log('Model changed to:', this.value);
            updateModelInfo();
            updateAvailableSizes();
        });
    }
    
    // Size change handler
    if (elements.imageSizeSelect) {
        elements.imageSizeSelect.addEventListener('change', function() {
            console.log('Size changed to:', this.value);
            updateSizePreview();
        });
    }
    
    // Style influence slider
    if (elements.styleInfluence) {
        elements.styleInfluence.addEventListener('input', function() {
            elements.styleInfluenceValue.textContent = this.value + '%';
        });
    }
    
    // Tab switching
    setupTabSwitching();
    
    // Theme card selection
    setupThemeCards();
    
    // Advanced settings toggle
    if (elements.advancedSettingsToggle) {
        elements.advancedSettingsToggle.addEventListener('click', function() {
            elements.advancedSettings.classList.toggle('active');
        });
    }
    
    // Boost prompt button
    if (elements.boostPromptBtn) {
        elements.boostPromptBtn.addEventListener('click', handleBoostPrompt);
    }
    
    // Generate button
    if (elements.generateBtn) {
        elements.generateBtn.addEventListener('click', handleGenerate);
    }
    
    // Action buttons
    if (elements.downloadBtn) {
        elements.downloadBtn.addEventListener('click', handleDownload);
    }
    
    if (elements.copyPromptBtn) {
        elements.copyPromptBtn.addEventListener('click', handleCopyPrompt);
    }
    
    if (elements.newImageBtn) {
        elements.newImageBtn.addEventListener('click', handleNewImage);
    }
    
    console.log('Event listeners set up successfully');
}

function setupTabSwitching() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(function(button) {
        button.addEventListener('click', function() {
            const tabId = this.getAttribute('data-tab');
            console.log('Tab clicked:', tabId);
            
            // Remove active class from all tabs
            tabButtons.forEach(function(btn) {
                btn.classList.remove('active');
            });
            tabContents.forEach(function(content) {
                content.classList.remove('active');
            });
            
            // Add active class to clicked tab
            this.classList.add('active');
            const targetTab = document.getElementById(tabId);
            if (targetTab) {
                targetTab.classList.add('active');
            }
        });
    });
}

function setupThemeCards() {
    const themeCards = document.querySelectorAll('.theme-card');
    
    themeCards.forEach(function(card) {
        card.addEventListener('click', function() {
            const prompt = this.getAttribute('data-prompt');
            console.log('Theme card clicked:', prompt);
            
            if (elements.promptInput) {
                elements.promptInput.value = prompt;
            }
            
            // Remove selected class from all cards
            themeCards.forEach(function(c) {
                c.classList.remove('selected');
            });
            
            // Add selected class to clicked card
            this.classList.add('selected');
        });
    });
}

function updateModelInfo() {
    if (!elements.modelSelect || !elements.modelInfo) return;
    
    const selectedModel = elements.modelSelect.value;
    const modelConfig = MODELS[selectedModel];
    
    console.log('Updating model info for:', selectedModel);
    
    if (modelConfig) {
        elements.modelInfo.innerHTML = 
            '<div class="model-info-text">' +
            '<strong>' + modelConfig.name + '</strong><br>' +
            modelConfig.description + '<br>' +
            'Max prompt: ' + modelConfig.max_prompt_length + ' chars' +
            '</div>';
    }
}

function updateSizePreview() {
    if (!elements.imageSizeSelect || !elements.sizePreview) return;
    
    const selectedSize = elements.imageSizeSelect.value;
    const sizeConfig = IMAGE_SIZES[selectedSize];
    
    console.log('Updating size preview for:', selectedSize);
    
    if (sizeConfig) {
        const iconClass = sizeConfig.icon === 'portrait' ? 'portrait' : 
                         (sizeConfig.icon === 'landscape' ? 'landscape' : '');
        
        elements.sizePreview.innerHTML = 
            '<div class="size-preview-icon ' + iconClass + '"></div>' +
            '<div class="size-preview-text">' + sizeConfig.width + ' × ' + sizeConfig.height + ' pixels</div>';
    }
}

function updateAvailableSizes() {
    if (!elements.modelSelect || !elements.imageSizeSelect) return;
    
    const selectedModel = elements.modelSelect.value;
    const modelConfig = MODELS[selectedModel];
    
    console.log('Updating available sizes for model:', selectedModel);
    
    if (modelConfig) {
        // Enable/disable size options based on model support
        Array.from(elements.imageSizeSelect.options).forEach(function(option) {
            const sizeKey = option.value;
            const isSupported = modelConfig.supports_sizes.includes(sizeKey);
            
            option.disabled = !isSupported;
            
            if (!isSupported) {
                option.classList.add('disabled-option');
            } else {
                option.classList.remove('disabled-option');
            }
            
            if (!isSupported && option.selected) {
                // Switch to first supported size
                const firstSupported = modelConfig.supports_sizes[0];
                elements.imageSizeSelect.value = firstSupported;
                updateSizePreview();
            }
        });
    }
}

function handleBoostPrompt() {
    console.log('Boost prompt clicked');
    
    const prompt = elements.promptInput.value.trim();
    
    if (!prompt) {
        showToast('Please enter a prompt first', 'error');
        return;
    }
    
    const originalButtonText = elements.boostPromptBtn.innerHTML;
    elements.boostPromptBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Enhancing...';
    elements.boostPromptBtn.disabled = true;
    
    fetch('/boost_prompt', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ prompt: prompt })
    })
    .then(function(response) {
        console.log('Boost prompt response:', response);
        if (!response.ok) {
            throw new Error('HTTP ' + response.status);
        }
        return response.json();
    })
    .then(function(data) {
        console.log('Boost prompt data:', data);
        if (data.boosted_prompt) {
            elements.promptInput.value = data.boosted_prompt;
            showToast('Prompt enhanced successfully', 'success');
        } else {
            showToast('Error: ' + (data.error || 'Unknown error'), 'error');
        }
    })
    .catch(function(error) {
        console.error('Boost prompt error:', error);
        showToast('Error enhancing prompt: ' + error.message, 'error');
    })
    .finally(function() {
        elements.boostPromptBtn.innerHTML = originalButtonText;
        elements.boostPromptBtn.disabled = false;
    });
}

function handleGenerate() {
    console.log('Generate clicked');
    
    const prompt = elements.promptInput.value.trim();
    
    if (!prompt) {
        showToast('Please enter a prompt', 'error');
        return;
    }
    
    const model = elements.modelSelect.value;
    const size = elements.imageSizeSelect.value;
    const style = elements.styleSelect.value;
    const styleInfluenceValue = parseInt(elements.styleInfluence.value);
    const addLogo = elements.addLogoCheckbox.checked;
    
    console.log('Generate params:', { model, size, style, styleInfluenceValue, addLogo });
    
    // Check prompt length
    const modelConfig = MODELS[model];
    if (prompt.length > modelConfig.max_prompt_length) {
        showToast('Prompt too long for ' + modelConfig.name + '. Maximum: ' + modelConfig.max_prompt_length + ' characters', 'error');
        return;
    }
    
    // Show loading state
    elements.previewPlaceholder.style.display = 'none';
    elements.previewImage.style.display = 'block';
    elements.loadingOverlay.style.display = 'flex';
    elements.previewDetails.classList.remove('active');
    
    fetch('/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            prompt: prompt,
            model: model,
            size: size,
            style: style,
            style_influence: styleInfluenceValue,
            add_logo: addLogo
        })
    })
    .then(function(response) {
        console.log('Generate response:', response);
        if (!response.ok) {
            throw new Error('HTTP ' + response.status);
        }
        return response.json();
    })
    .then(function(data) {
        console.log('Generate data:', data);
        if (data.success && data.image) {
            // Display the generated image
            elements.generatedImage.src = 'data:image/png;base64,' + data.image;
            elements.usedPrompt.textContent = data.prompt;
            elements.usedModelSize.textContent = data.model + ' • ' + data.size;
            
            // Hide loading overlay
            elements.loadingOverlay.style.display = 'none';
            elements.previewDetails.classList.add('active');
            
            // Store filename for download
            elements.downloadBtn.setAttribute('data-filename', data.filename);
            
            showToast('Image generated successfully', 'success');
        } else {
            showToast(data.error || 'Failed to generate image', 'error');
            elements.previewPlaceholder.style.display = 'flex';
            elements.previewImage.style.display = 'none';
        }
    })
    .catch(function(error) {
        console.error('Generate error:', error);
        showToast('Error generating image: ' + error.message, 'error');
        elements.previewPlaceholder.style.display = 'flex';
        elements.previewImage.style.display = 'none';
    });
}

function handleDownload() {
    console.log('Download clicked');
    
    const imageData = elements.generatedImage.src;
    const filename = elements.downloadBtn.getAttribute('data-filename') || 'generated-image.png';
    
    const link = document.createElement('a');
    link.href = imageData;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    showToast('Image downloaded', 'success');
}

function handleCopyPrompt() {
    console.log('Copy prompt clicked');
    
    const promptText = elements.usedPrompt.textContent;
    navigator.clipboard.writeText(promptText).then(function() {
        showToast('Prompt copied to clipboard', 'success');
    }).catch(function(err) {
        console.error('Failed to copy prompt:', err);
        showToast('Failed to copy prompt', 'error');
    });
}

function handleNewImage() {
    console.log('New image clicked');
    
    elements.previewPlaceholder.style.display = 'flex';
    elements.previewImage.style.display = 'none';
    elements.previewDetails.classList.remove('active');
    elements.promptInput.value = '';
    
    // Remove selected class from theme cards
    const themeCards = document.querySelectorAll('.theme-card');
    themeCards.forEach(function(card) {
        card.classList.remove('selected');
    });
}

function showToast(message, type) {
    type = type || 'success';
    console.log('Toast:', message, type);
    
    elements.toast.className = 'toast';
    elements.toast.classList.add(type);
    elements.toast.classList.add('show');
    
    const icon = elements.toast.querySelector('i');
    if (type === 'success') {
        icon.className = 'fas fa-check-circle success';
    } else {
        icon.className = 'fas fa-exclamation-circle error';
    }
    
    elements.toastMessage.textContent = message;
    
    setTimeout(function() {
        elements.toast.classList.remove('show');
    }, 3000);
}
