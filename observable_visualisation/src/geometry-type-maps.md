---
title: Napoleonic Cadaster - Average casa size
toc: false
style: components/custom-style.css
---


```js
// Explicit import of leaflet to avoid issues with the Leaflet.heat plugin
import L from "npm:leaflet";
```

```js
// Wait for L to be defined before importing the Leaflet.heat plugin
// This is necessary because Leaflet.heat depends on the L variable being defined
if (L === undefined) console.error("L is undefined");

// Leaflet.heat: https://github.com/Leaflet/Leaflet.heat/
import "./plugins/leaflet-heat.js";
import {createParishGeometryTypeMap, createGeometryTypeColoredMap } from "./components/map-geometry-type.js";
```

# Type of Geometry.
On the cadastral map, each delineated shape was filled with a specific color indicating the type of cadastral object it represents. This graphic design was expressly indicated in the [Recueil Methodique](https://gallica.bnf.fr/ark:/12148/bpt6k96475008.texteImage), which all administrations under French administration were required to follow.

Buildings are rendered in pink, water (canals) in blue, subporches in orange, streets in white, and courtyards in beige. During the digitization of the cadastral register, this typological information was embedded directly within the geometric features, enabling the creation of a faithful digital twin of the original cadastral map.


```js
const parishData = FileAttachment("./data/1740_redrawn_parishes_cleaned_wikidata_standardised.geojson").json();
const parcelData = FileAttachment("./data/venice_1808_landregister_geometries.geojson").json();
```


<!-- Create the map container -->
<div id="map-container-geo-type" class="map-component"></div>

```js
// Call the creation function and store the results
const allGeoTypeMapComponents = createGeometryTypeColoredMap("map-container-geo-type", parcelData);
```

_This is also handy in order to compute statistics regarding urban occupation of the city of venice. Select the geometry type from the dropdown menue below to display the total surface of this geometry type, as well as a map and ranking of the prevalence of such a data in the different parish zones of Venice._

<select id="selector-geo-type"></select>

<strong>
    <div id="selected-geotype-total-value"></div>
</strong>
<!-- Create the map container -->

<div id="map-container-parish-geo-type" class="map-component"></div>
<!-- Create the tanble container -->
<div class="block-container">
    <div id="table-container-parish-geo-type-perc-ranking">
    </div>
</div>


```js
const selector = document.getElementById("selector-geo-type");
//Create array of options to be added
const geoTypes = ["building", "street", "courtyard", "sottoportico"];

//Create and append the options
for (var i = 0; i < geoTypes.length; i++) {
    const option = document.createElement("option");
    option.value = geoTypes[i];
    option.text = geoTypes[i];
    selector.appendChild(option);
}

let map = L.map("map-container-parish-geo-type", {minZoom: 0, maxZoom:18}).setView([45.4382745, 12.3433387 ], 14);

function generateParishMapFromGeometryMapSelection(){
    // uninitializing all data. 
    map.off();
    map.remove();
    const mapContainer = document.getElementById("map-container-parish-geo-type");
    mapContainer.innerHTML = "";
    map = L.map("map-container-parish-geo-type", {minZoom: 0, maxZoom:18}).setView([45.4382745, 12.3433387 ], 14); 
    const totalSurfaceContainer = document.getElementById("selected-geotype-total-value");
    totalSurfaceContainer.innerHTML = "";
    const tableContainer = document.getElementById("table-container-parish-geo-type-perc-ranking");
    tableContainer.textContent = "";
    tableContainer.innerHTLM = "";

    const selector = document.getElementById("selector-geo-type");
    const geoTypeSelected = selector.options[selector.selectedIndex].text;
    // Call the creation function and store the results
    const geoTypeMapComponents = createParishGeometryTypeMap(map, parcelData, parishData, geoTypeSelected);

    // affecting values to the window is the easiest way to break the observable sandbox and make code available in the plain JS context of the webpage.
    window.highlightFeature = (name) => {
        geoTypeMapComponents.geoJsonLayerParish.resetStyle();
        // for some reason, observable does not let me set intermediat variable, so all action on layer has to call the layer from the hashMap again.
        geoTypeMapComponents.map.flyTo(geoTypeMapComponents.parishNameLayerMap.get(name).getBounds().getCenter(), 15.4);
        geoTypeMapComponents.parishNameLayerMap.get(name).setStyle({
            weight: 5,
            color: '#FFF',
            dashArray: '',
            fillOpacity: 0.7
        });
        geoTypeMapComponents.parishNameLayerMap.get(name).bringToFront();
        geoTypeMapComponents.parishNameLayerMap.get(name).openPopup();
    };

    document.getElementById("selected-geotype-total-value").innerHTML = "Total surface of " + geoTypeSelected+ ":"  + String(geoTypeMapComponents.totalSurface.toFixed(2))+"m2";

    const table = Inputs.table(geoTypeMapComponents.tableData, {
        header: {
            name: "Parish Name",
            geotype_percentage: `Percentage of ${geoTypeSelected} type surface per parish`,
        },
        format: {
            name: id => htl.html`<a class="hover-line table-row-padding" onclick=window.highlightFeature("${id}");>${id}</a>`,
        geotype_percentage: x => String((x*100.0).toFixed(2))+'%'
        }, 
        select: false
    });
    tableContainer.append(table);
}

generateParishMapFromGeometryMapSelection();
selector.onchange = generateParishMapFromGeometryMapSelection;
```