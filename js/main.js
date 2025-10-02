window.onload = function() {
    initWindow();
    make_carousel('results-edit2', edit_carousel_item_template, edit_items, 2, 2);
    make_carousel('results-variants', variants_carousel_item_template, variants_items, 2, 1);
    make_carousel('results-manipulation', manipulation_carousel_item_template, manipulation_items, 1, 1);
    make_selection_panel('results-scene', scene_selection_panel_thumbnail_template, scene_selection_panel_item_template, scene_items);
};
