var edit_items = [
    // boiler_19
    { video: "flux_edit/boiler_19/ori.mp4", video2: "flux_edit/boiler_19/vox.mp4", video3: "flux_edit/boiler_19/edit.mp4", prompt: "flux_edit/boiler_19/cond_img.png", model: "flux_edit/boiler_19/ori.glb", model2: "flux_edit/boiler_19/vox.glb", model3: "flux_edit/boiler_19/edit.glb", source: "DALL-E 3", alt: "boiler_19", text: "Add a conical top to the cylinder"},
    // reptile_5
    { video: "flux_edit/reptile_5/ori.mp4", video2: "flux_edit/reptile_5/vox.mp4", video3: "flux_edit/reptile_5/edit.mp4", prompt: "flux_edit/reptile_5/cond_img.png", model: "flux_edit/reptile_5/ori.glb", model2: "flux_edit/reptile_5/vox.glb", model3: "flux_edit/reptile_5/edit.glb", source: "DALL-E 3", alt: "reptile_5", text: "Add a pair of wings to the back"},
    // dive_7
    { video: "flux_edit/dive_7/ori.mp4", video2: "flux_edit/dive_7/vox.mp4", video3: "flux_edit/dive_7/edit.mp4", prompt: "flux_edit/dive_7/cond_img.png", model: "flux_edit/dive_7/ori.glb", model2: "flux_edit/dive_7/vox.glb", model3: "flux_edit/dive_7/edit.glb", source: "DALL-E 3", alt: "dive_7", text: "Add a periscope to the top front"},
    // kneel_11
    { video: "flux_edit/kneel_11/ori.mp4", video2: "flux_edit/kneel_11/vox.mp4", video3: "flux_edit/kneel_11/edit.mp4", prompt: "flux_edit/kneel_11/cond_img.png", model: "flux_edit/kneel_11/ori.glb", model2: "flux_edit/kneel_11/vox.glb", model3: "flux_edit/kneel_11/edit.glb", source: "DALL-E 3", alt: "kneel_11", text: "Add a decorative shield to the knight&#39;s back"},
    // coffin_11
    { video: "flux_edit/coffin_11/ori.mp4", video2: "flux_edit/coffin_11/vox.mp4", video3: "flux_edit/coffin_11/edit.mp4", prompt: "flux_edit/coffin_11/cond_img.png", model: "flux_edit/coffin_11/ori.glb", model2: "flux_edit/coffin_11/vox.glb", model3: "flux_edit/coffin_11/edit.glb", source: "DALL-E 3", alt: "coffin_11", text: "Remove the golden head from the sarcophagus"},
    // individual_5
    { video: "flux_edit/individual_5/ori.mp4", video2: "flux_edit/individual_5/vox.mp4", video3: "flux_edit/individual_5/edit.mp4", prompt: "flux_edit/individual_5/cond_img.png", model: "flux_edit/individual_5/ori.glb", model2: "flux_edit/individual_5/vox.glb", model3: "flux_edit/individual_5/edit.glb", source: "DALL-E 3", alt: "individual_5", text: "Replace the black shirt with a white button-up shirt"},
    // kitchen window_6
    { video: "flux_edit/kitchen window_6/ori.mp4", video2: "flux_edit/kitchen window_6/vox.mp4", video3: "flux_edit/kitchen window_6/edit.mp4", prompt: "flux_edit/kitchen window_6/cond_img.png", model: "flux_edit/kitchen window_6/ori.glb", model2: "flux_edit/kitchen window_6/vox.glb", model3: "flux_edit/kitchen window_6/edit.glb", source: "DALL-E 3", alt: "kitchen window_6", text: "Add a decorative arch above the window frame"},
    // launch_17
    { video: "flux_edit/launch_17/ori.mp4", video2: "flux_edit/launch_17/vox.mp4", video3: "flux_edit/launch_17/edit.mp4", prompt: "flux_edit/launch_17/cond_img.png", model: "flux_edit/launch_17/ori.glb", model2: "flux_edit/launch_17/vox.glb", model3: "flux_edit/launch_17/edit.glb", source: "DALL-E 3", alt: "launch_17", text: "Add a small, rectangular lifeboat at the stern"},
    // leprechaun_2
    { video: "flux_edit/leprechaun_2/ori.mp4", video2: "flux_edit/leprechaun_2/vox.mp4", video3: "flux_edit/leprechaun_2/edit.mp4", prompt: "flux_edit/leprechaun_2/cond_img.png", model: "flux_edit/leprechaun_2/ori.glb", model2: "flux_edit/leprechaun_2/vox.glb", model3: "flux_edit/leprechaun_2/edit.glb", source: "DALL-E 3", alt: "leprechaun_2", text: "Add a golden scepter to his hand"},
    // lizard_19
    { video: "flux_edit/lizard_19/ori.mp4", video2: "flux_edit/lizard_19/vox.mp4", video3: "flux_edit/lizard_19/edit.mp4", prompt: "flux_edit/lizard_19/cond_img.png", model: "flux_edit/lizard_19/ori.glb", model2: "flux_edit/lizard_19/vox.glb", model3: "flux_edit/lizard_19/edit.glb", source: "DALL-E 3", alt: "lizard_19", text: "Add a small, pointed horn above the lizard&#39;s head"},
    // macaw_3
    { video: "flux_edit/macaw_3/ori.mp4", video2: "flux_edit/macaw_3/vox.mp4", video3: "flux_edit/macaw_3/edit.mp4", prompt: "flux_edit/macaw_3/cond_img.png", model: "flux_edit/macaw_3/ori.glb", model2: "flux_edit/macaw_3/vox.glb", model3: "flux_edit/macaw_3/edit.glb", source: "DALL-E 3", alt: "macaw_3", text: "Add a small, yellow beak to the parrot&#39;s head"},
    // philosopher_5
    { video: "flux_edit/philosopher_5/ori.mp4", video2: "flux_edit/philosopher_5/vox.mp4", video3: "flux_edit/philosopher_5/edit.mp4", prompt: "flux_edit/philosopher_5/cond_img.png", model: "flux_edit/philosopher_5/ori.glb", model2: "flux_edit/philosopher_5/vox.glb", model3: "flux_edit/philosopher_5/edit.glb", source: "DALL-E 3", alt: "philosopher_5", text: "Add a long, flowing beard"},
    // railcar_16
    { video: "flux_edit/railcar_16/ori.mp4", video2: "flux_edit/railcar_16/vox.mp4", video3: "flux_edit/railcar_16/edit.mp4", prompt: "flux_edit/railcar_16/cond_img.png", model: "flux_edit/railcar_16/ori.glb", model2: "flux_edit/railcar_16/vox.glb", model3: "flux_edit/railcar_16/edit.glb", source: "DALL-E 3", alt: "railcar_16", text: "Replace the wheels with larger, more robust ones"},
    
    // tavern_7
    { video: "flux_edit/tavern_7/ori.mp4", video2: "flux_edit/tavern_7/vox.mp4", video3: "flux_edit/tavern_7/edit.mp4", prompt: "flux_edit/tavern_7/cond_img.png", model: "flux_edit/tavern_7/ori.glb", model2: "flux_edit/tavern_7/vox.glb", model3: "flux_edit/tavern_7/edit.glb", source: "DALL-E 3", alt: "tavern_7", text: "Add a small balcony above the arched doorway"},
    // tourist attraction_17
    { video: "flux_edit/tourist attraction_17/ori.mp4", video2: "flux_edit/tourist attraction_17/vox.mp4", video3: "flux_edit/tourist attraction_17/edit.mp4", prompt: "flux_edit/tourist attraction_17/cond_img.png", model: "flux_edit/tourist attraction_17/ori.glb", model2: "flux_edit/tourist attraction_17/vox.glb", model3: "flux_edit/tourist attraction_17/edit.glb", source: "DALL-E 3", alt: "tourist attraction_17", text: "Remove the lion&#39;s mane"},
    // turtle_19
    { video: "flux_edit/turtle_19/ori.mp4", video2: "flux_edit/turtle_19/vox.mp4", video3: "flux_edit/turtle_19/edit.mp4", prompt: "flux_edit/turtle_19/cond_img.png", model: "flux_edit/turtle_19/ori.glb", model2: "flux_edit/turtle_19/vox.glb", model3: "flux_edit/turtle_19/edit.glb", source: "DALL-E 3", alt: "turtle_19", text: "Add a small, triangular fin to the turtle&#39;s tail"},
];


function edit_carousel_item_template(item) {
    return `<div class="x-card clickable" style="min-width: 120px" onclick=\'openWindow(edit_window_template(${JSON.stringify(item)}))\'>
                <div class="x-labels">
                    <div class="x-label">GLB ✓</div>
                </div>
                <div style="width: 100%; display: flex; gap: 8px;">
                    <div style="flex: 1; aspect-ratio: 1;">
                        <video autoplay playsinline loop muted
                             style="width: 100%; height: 100%; object-fit: cover;"
                             src="assets_glb/${item.video}"></video>
                    </div>
                    <div style="flex: 1; aspect-ratio: 1;">
                        <video autoplay playsinline loop muted
                             style="width: 100%; height: 100%; object-fit: cover;"
                             src="assets_glb/${item.video2}"></video>
                    </div>
                    <div style="flex: 1; aspect-ratio: 1;">
                        <video autoplay playsinline loop muted
                             style="width: 100%; height: 100%; object-fit: cover;"
                             src="assets_glb/${item.video3}"></video>
                    </div>
                </div>
                <div class="caption">
                    <div class="x-image-prompt">
                        <img src="assets_glb/${item.prompt}" alt="${item.alt}">
                    </div>
                    <div class="x-handwriting" style="text-align: center; margin-top: 10px; margin-bottom: 10px;">
                        ${item.text}
                    </div>
                </div>
            </div>`;
}

function edit_window_template(item) {
    let prompt = `<div class="x-image-prompt"><img src="assets_glb/${item.prompt}" alt="${item.alt}"></div>`;
    let text = `<div class="x-handwriting">${item.text}</div>`;
    let panel = edit_asset_panel_template(prompt, text);
    item = JSON.parse(JSON.stringify(item));
    item.model = 'assets_glb/' + item.model
    item.model2 = 'assets_glb/' + item.model2
    item.model3 = 'assets_glb/' + item.model3
    return edit_modelviewer_window_template(item, panel);
}
