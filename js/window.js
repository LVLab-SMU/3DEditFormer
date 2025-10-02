var fullscreenElement = null;


function preventDefault(e) {
    e.preventDefault();
}


function stopPropagation(e) {
    e.stopPropagation();
}


function initWindow() {
    fullscreenElement = document.createElement('div');
    fullscreenElement.id = 'fullscreen';
    fullscreenElement.innerHTML = `
        <div id="window">
            <div id="close" onclick="closeWindow()">✕</div>
            <div id="content"></div>
        </div>
    `;
    document.body.appendChild(fullscreenElement);
    fullscreenElement.addEventListener('wheel', preventDefault);
    fullscreenElement.addEventListener('touchmove', preventDefault);
}


function openWindow(content) {
    let contentElement = fullscreenElement.querySelector('#content');
    contentElement.innerHTML = content;
    fullscreenElement.style.display = 'flex';
    setTimeout(() => {
        fullscreenElement.style.opacity = 1;
    }, 100);
    
    // if copntent height is greater than window height, enable scroll
    let contentHeight = contentElement.clientHeight;
    let innerHeight = contentElement.children[0].clientHeight;
    if (innerHeight > contentHeight) {
        contentElement.addEventListener('wheel', stopPropagation);
        contentElement.addEventListener('touchmove', stopPropagation);
    }
    else {
        contentElement.removeEventListener('wheel', stopPropagation);
        contentElement.removeEventListener('touchmove', stopPropagation);
    }
}


function closeWindow() {
    window_state = {};
    fullscreenElement.style.opacity = 0;
    setTimeout(() => {
        fullscreenElement.style.display = 'none';
    }, 250);
}


var window_state = {};


function hideTexture() {
    let appearanceButton = document.getElementById('appearance-button');
    let geometryButton = document.getElementById('geometry-button');
    appearanceButton.classList.remove('checked');
    geometryButton.classList.add('checked');
    let modelViewer = document.getElementById('modelviewer');
    if (modelViewer.model.materials[0].pbrMetallicRoughness.baseColorTexture.texture === null) return;
    window_state.textures = [];
    for (let i = 0; i < modelViewer.model.materials.length; i++) {
        window_state.textures.push(modelViewer.model.materials[i].pbrMetallicRoughness.baseColorTexture.texture);
    }
    window_state.exposure = modelViewer.exposure;
    modelViewer.environmentImage = 'assets_glb/env_maps/gradient.jpg';
    for (let i = 0; i < modelViewer.model.materials.length; i++) {
        modelViewer.model.materials[i].pbrMetallicRoughness.baseColorTexture.setTexture(null);
    }
    modelViewer.exposure = 5;

    let modelViewer2 = document.getElementById('modelviewer2');
    if (modelViewer2.model.materials[0].pbrMetallicRoughness.baseColorTexture.texture === null) return;
    window_state.textures2 = [];
    for (let i = 0; i < modelViewer2.model.materials.length; i++) {
        window_state.textures2.push(modelViewer2.model.materials[i].pbrMetallicRoughness.baseColorTexture.texture);
    }
    window_state.exposure = modelViewer2.exposure;
    modelViewer2.environmentImage = 'assets_glb/env_maps/gradient.jpg';
    for (let i = 0; i < modelViewer2.model.materials.length; i++) {
        modelViewer2.model.materials[i].pbrMetallicRoughness.baseColorTexture.setTexture(null);
    }
    modelViewer2.exposure = 5;

    let modelViewer3 = document.getElementById('modelviewer3');
    if (modelViewer3.model.materials[0].pbrMetallicRoughness.baseColorTexture.texture === null) return;
    window_state.textures3 = [];
    for (let i = 0; i < modelViewer3.model.materials.length; i++) {
        window_state.textures3.push(modelViewer3.model.materials[i].pbrMetallicRoughness.baseColorTexture.texture);
    }
    window_state.exposure = modelViewer3.exposure;
    modelViewer3.environmentImage = 'assets_glb/env_maps/gradient.jpg';
    for (let i = 0; i < modelViewer3.model.materials.length; i++) {
        modelViewer3.model.materials[i].pbrMetallicRoughness.baseColorTexture.setTexture(null);
    }
    modelViewer3.exposure = 5;
}


function showTexture() {
    let appearanceButton = document.getElementById('appearance-button');
    let geometryButton = document.getElementById('geometry-button');
    appearanceButton.classList.add('checked');
    geometryButton.classList.remove('checked');
    let modelViewer = document.getElementById('modelviewer');
    if (modelViewer.model.materials[0].pbrMetallicRoughness.baseColorTexture.texture !== null) return;
    modelViewer.environmentImage = 'assets_glb/env_maps/white.jpg';
    for (let i = 0; i < modelViewer.model.materials.length; i++) {
        modelViewer.model.materials[i].pbrMetallicRoughness.baseColorTexture.setTexture(window_state.textures[i]);
    }
    modelViewer.exposure = window_state.exposure;

    let modelViewer2 = document.getElementById('modelviewer2');
    if (modelViewer2.model.materials[0].pbrMetallicRoughness.baseColorTexture.texture !== null) return;
    modelViewer2.environmentImage = 'assets_glb/env_maps/white.jpg';
    for (let i = 0; i < modelViewer2.model.materials.length; i++) {
        modelViewer2.model.materials[i].pbrMetallicRoughness.baseColorTexture.setTexture(window_state.textures2[i]);
    }
    modelViewer2.exposure = window_state.exposure;

    let modelViewer3 = document.getElementById('modelviewer3');
    if (modelViewer3.model.materials[0].pbrMetallicRoughness.baseColorTexture.texture !== null) return;
    modelViewer3.environmentImage = 'assets_glb/env_maps/white.jpg';
    for (let i = 0; i < modelViewer3.model.materials.length; i++) {
        modelViewer3.model.materials[i].pbrMetallicRoughness.baseColorTexture.setTexture(window_state.textures3[i]);
    }
    modelViewer3.exposure = window_state.exposure;
}


function showAnnotations() {
    let showButton = document.getElementById('show-annotations-button');
    let hideButton = document.getElementById('hide-annotations-button');
    showButton.classList.add('checked');
    hideButton.classList.remove('checked');
    let modelViewer = document.getElementById('modelviewer');
    let annotations = modelViewer.querySelectorAll('button');
    for (let i = 0; i < annotations.length; i++) {
        annotations[i].style.display = 'block';
    }
    setTimeout(() => {
        for (let i = 0; i < annotations.length; i++) {
            annotations[i].style.opacity = 1;
        }
    }, 100);
}


function hideAnnotations() {
    let showButton = document.getElementById('show-annotations-button');
    let hideButton = document.getElementById('hide-annotations-button');
    showButton.classList.remove('checked');
    hideButton.classList.add('checked');
    let modelViewer = document.getElementById('modelviewer');
    let annotations = modelViewer.querySelectorAll('button');
    for (let i = 0; i < annotations.length; i++) {
        annotations[i].style.opacity = 0;
    }
    setTimeout(() => {
        for (let i = 0; i < annotations.length; i++) {
            annotations[i].style.display = 'none';
        }
    }, 200);
}


function lookAtAsset(index) {
    let modelViewer = document.getElementById('modelviewer');
    let viewAssetButton = document.getElementById('view-asset-button');
    let assetList = document.getElementById('asset-list');
    let hasChecked = assetList.children[index].classList.contains('checked');
    if (hasChecked) {
        for (let i = 0; i < window_state.assets.length; i++) {
            assetList.children[i].classList.remove('checked');
        }
        modelViewer.cameraTarget = 'none';
        viewAssetButton.classList.remove('enabled');
        viewAssetButton.classList.add('disabled');
        viewAssetButton.onclick = null;
    }
    else {
        for (let i = 0; i < window_state.assets.length; i++) {
            if (i === index) {
                assetList.children[i].classList.add('checked');
            }
            else {
                assetList.children[i].classList.remove('checked');
            }
        }
        modelViewer.cameraTarget = window_state.assets[index].position.map(p => p + 'm').join(' ');
        viewAssetButton.classList.remove('disabled');
        viewAssetButton.classList.add('enabled');
        viewAssetButton.onclick = () => viewAsset(index);
    }
}


function viewAsset(index) {
    hideAnnotations();
    showTexture();

    let modelViewer = document.getElementById('modelviewer');
    let viewAssetButton = document.getElementById('view-asset-button');
    let assetList = document.getElementById('asset-list');

    document.getElementById('scene-mode-desc').style.display = 'none';
    document.getElementById('asset-mode-desc').style.display = 'block';
    document.getElementById('prompt-group').style.display = 'block';
    document.querySelector('.modelviewer-panel-prompt').innerHTML = window_state.prompt_template(window_state.assets[index]);
    document.getElementById('annotations-group').style.display = 'none';

    if (!window_state.scene_src) {
        window_state.scene_src = modelViewer.src;
    }
    modelViewer.cameraTarget = 'none';
    modelViewer.src = window_state.assets[index].model;

    for (let i = 0; i < window_state.assets.length; i++) {
        if (i === index) {
            assetList.children[i].classList.add('checked');
        }
        else {
            assetList.children[i].classList.remove('checked');
        }
        assetList.children[i].onclick = () => viewAsset(i);
    }

    viewAssetButton.innerHTML = 'Back to Scene';
    viewAssetButton.onclick = backToScene;
}


function backToScene() {
    showTexture();

    let modelViewer = document.getElementById('modelviewer');
    let viewAssetButton = document.getElementById('view-asset-button');
    let assetList = document.getElementById('asset-list');

    document.getElementById('scene-mode-desc').style.display = 'block';
    document.getElementById('asset-mode-desc').style.display = 'none';
    document.getElementById('prompt-group').style.display = 'none';
    document.getElementById('annotations-group').style.display = 'block';

    modelViewer.cameraTarget = 'none';
    modelViewer.src = window_state.scene_src;
    window_state.scene_src = null;

    for (let i = 0; i < window_state.assets.length; i++) {
        assetList.children[i].classList.remove('checked');
        assetList.children[i].onclick = () => lookAtAsset(i);
    }

    viewAssetButton.innerHTML = 'View Asset';
    viewAssetButton.classList.remove('enabled');
    viewAssetButton.classList.add('disabled');
    viewAssetButton.onclick = null;
}


function downloadGLB() {
    let modelViewer = document.getElementById('modelviewer');
    window.open(modelViewer.src);
}


function edit_asset_panel_template(prompt, text) {
    return `
        <div class="x-section-title small" style="display:flex; align-items:center; gap:1.0em;">
            <div class="x-gradient-font">Prompt</div>
            <span>${text}</span>
        </div>
        <div class="modelviewer-panel-prompt">
            ${prompt}
        </div>
        <div class="x-section-title small"><div class="x-gradient-font">Display Mode</div></div>
        <div class="x-left-align">
            <div id="appearance-button" class="modelviewer-panel-button small checked" onclick="showTexture()">Appearance</div>
            <div id="geometry-button" class="modelviewer-panel-button small" onclick="hideTexture()">Geometry</div>
        </div>
        <div class="x-flex-spacer"></div>
    `;
}

function edit_modelviewer_window_template(item, panel, config) {
    let viewer_size = config && config.viewer_size || 500;
    let panel_size = config && config.panel_size || 300;
    let show_annotations = config && config.show_annotations || false;
    html = `<div class="x-row" style="align-items: stretch; flex-wrap: wrap; width: ${viewer_size + viewer_size + panel_size + 64}px; max-width: calc(100vw - 32px);">
                <div class="modelviewer-container" style="width: ${viewer_size * 0.65}px;">
                    <div class="x-section-title small" style="display:flex; justify-content:center; align-items:center; width:100%;">
                        <div class="x-gradient-font">Source 3D</div>
                    </div>
                    <model-viewer
                        id="modelviewer"
                        src="${item.model}"
                        camera-controls
                        tone-mapping="natural"
                        shadow-intensity="1"
                        environment-image="assets_glb/env_maps/white.jpg"
                        exposure="${item.exposure || 5}"
                        >
                    </model-viewer>
                </div>
                <div class="modelviewer-container" style="width: ${viewer_size * 0.65}px;">
                    <div class="x-section-title small" style="display:flex; justify-content:center; align-items:center; width:100%;">
                        <div class="x-gradient-font">VoxHammer</div>
                    </div>
                    <model-viewer
                        id="modelviewer2"
                        src="${item.model2}"
                        camera-controls
                        tone-mapping="natural"
                        shadow-intensity="1"
                        environment-image="assets_glb/env_maps/white.jpg"
                        exposure="${item.exposure || 5}"
                        >
                    </model-viewer>
                </div>
                <div class="modelviewer-container" style="width: ${viewer_size * 0.65}px;">
                    <div class="x-section-title small" style="display:flex; justify-content:center; align-items:center; width:100%;">
                        <div class="x-gradient-font">3DEditFormer (Ours)</div>
                    </div>
                    <model-viewer
                        id="modelviewer3"
                        src="${item.model3}"
                        camera-controls
                        tone-mapping="natural"
                        shadow-intensity="1"
                        environment-image="assets_glb/env_maps/white.jpg"
                        exposure="${item.exposure || 5}"
                        >`
    if (show_annotations) {
        window_state.assets = item.assets;
        window_state.prompt_template = item.prompt_template;
        for (let i = 0; i < item.assets.length; i++) {
            html += `<button slot="hotspot-${i}" data-position="${item.assets[i].position.join(' ')}">${item.assets[i].name}</button>`;
        }
    }
    html += `        </model-viewer>
                </div>
                <div class="modelviewer-panel" style="flex: 1 1 ${panel_size}px;">
                    ${panel}
                </div>
            </div>`;
    return html;
}
