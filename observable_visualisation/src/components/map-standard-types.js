// Explicit import of leaflet to avoid issues with the Leaflet.heat plugin
import L from "npm:leaflet";
import {geometryRegistryMap, registryListToHTML, genereateBaseSommarioniBgLayers, cleanStdVal} from "./common.js";

if (L === undefined) console.error("L is undefined");

// Leaflet.heat: https://github.com/Leaflet/Leaflet.heat/
import "../plugins/leaflet-heat.js";



// Create Map and Layer - Runs Once
export function createMapAndLayers(mapContainer, geojsonData, registryData, registryField, enabledLayer) {
    const map = L.map(mapContainer, {minZoom: 0, maxZoom:18}).setView([45.4382745, 12.3433387 ], 14);

    // this allows to get the current selected overlays from the control 
    L.Control.Layers.include({
        getOverlays: function() {
            // create hash to hold all layers
            var control, layers;
            layers = {};
            control = this;

            // loop thru all layers in control
            control._layers.forEach(function(obj) {
            var layerName;

            // check if layer is an overlay
            if (obj.overlay) {
                // get name of overlay
                layerName = obj.name;
                // store whether it's present on the map or not
                return layers[layerName] = control._map.hasLayer(obj.layer);
            }
            });

            return layers;
        }
    });
    // Crate a control to switch between layers
    const layerControl = L.control.layers().addTo(map);


    // Add all default layers to the map.
    const bgLayerList = genereateBaseSommarioniBgLayers();
    for( let [key, value] of Object.entries(bgLayerList)){
        layerControl.addBaseLayer(value, key);
    } 
    bgLayerList["Cadastral Board"].addTo(map);
    let registryMap = geometryRegistryMap(registryData);
    //filtering the data to keep only geometries referenced in the registry (i.e. the ones having a geometry_id value)
    let feats = geojsonData.features.filter(feature => feature.properties.geometry_id && feature.properties.parcel_number);
    // then fetching the value of "ownership_types" from the registry and adding them to the properties of the features
    geojsonData.features = feats.map(feature => {
        const geometry_id = String(feature.properties.geometry_id);
        const registryEntries = registryMap.get(geometry_id);
        if (registryEntries) {
            let values = [];
            registryEntries.forEach(entry => {
                if (entry[registryField]) {
                    if (typeof entry[registryField] === "string") {
                        values.push(cleanStdVal(entry[registryField]));
                    } else if (Array.isArray(entry[registryField])) {
                        let vals = entry[registryField];
                        for (let i = 0; i < vals.length; i++) {
                            values.push(cleanStdVal(vals[i]));
                        }
                    }
                }
            });
            // Remove duplicates
            values = [...new Set(values)];
            // Add the registryFields to the feature properties
            feature.properties[registryField] = values;
        }
        return feature;
    }).filter(feature => feature.properties[registryField]);
    let mapLayerGroups = {};
    // pop up needs to be generated dyinamically based on the current selected standard value, to only display registry entries that match the current selected standard value
    function onPopupClick(e) {
        // Get the clicked feature layer
        const featureLayer = e.target;
        // Get the geometry_id from the feature properties
        const geometryId = String(featureLayer.feature.properties.geometry_id);
        const allRegistryEntries = registryMap.get(geometryId);
        const currSelectedStdValues = Object.entries(layerControl.getOverlays()).filter(sel => sel[1]).map(v => v[0]);

        function filterEntrysByStdValue(entry) {
            if (entry === undefined || entry[registryField] === undefined || entry[registryField] === null) {
                return currSelectedStdValues.includes("0 values");
            }
            if (typeof entry[registryField] === "string") {
                if (entry[registryField] === "") {
                    return currSelectedStdValues.includes("0 values");
                }
                return currSelectedStdValues.includes(cleanStdVal(entry[registryField]));
            } else if (Array.isArray(entry[registryField])) {
                if (entry[registryField].length === 0 || (entry[registryField].length === 1 && entry[registryField][0] === "")) {
                    return currSelectedStdValues.includes("0 values");
                }
                for (let i = 0; i < entry[registryField].length; i++) {
                    if (currSelectedStdValues.includes(cleanStdVal(entry[registryField][i]))) {
                        return true;
                    }
                }
            }
            return false;
        }
        // for debugging purposes
        // const wayId = featureLayer.feature.properties.id;

        let html = registryListToHTML(allRegistryEntries.filter(filterEntrysByStdValue))
        // Add a popup to the feature layer
        e.target.bindPopup(html, {'maxWidth':'350','maxHeight':'500','minWidth':'150'}).openPopup();
    }

    function onEachFeature(feature, featureLayer) {

        //does layerGroup already exist? if not create it and add to map
        let values = feature.properties[registryField];
        if(values.length === 0){
            values = [""];
        }
        for (let i = 0; i < values.length; i++) {
            let value = values[i];
            if (value === ""){
                value = "0 values";
            }
            var lg = mapLayerGroups[value];

            if (lg === undefined) {
                lg = new L.layerGroup();
                //add the layer to the map
                if (enabledLayer){
                    lg.addTo(map);
                }
                //store layer
                mapLayerGroups[value] = lg;
            }

            lg.addLayer(featureLayer);
        }    
        featureLayer.on('click', onPopupClick);
    }
    // Store map from geom_id -> leaflet layer instance
    const featureLayersMap = new Map();
    const geoJsonLayer = L.geoJSON(geojsonData, {onEachFeature: onEachFeature});
    for (const [key, value] of Object.entries(mapLayerGroups).sort((a, b) => a[0].localeCompare(b[0]))) {
        layerControl.addOverlay(value, key);
    }

    // Return the the map instance, the layer group, and the mapping
    return { map, layerControl, geoJsonLayer, featureLayersMap, mapLayerGroups };
}
