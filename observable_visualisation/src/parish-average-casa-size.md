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
import {createParishCasaAverageSurfaceHeatMap } from "./components/map-parish.js";
```

# Average surface of “house” function per parish.

```js
const parishData = FileAttachment("./data/1740_redrawn_parishes_cleaned_wikidata_standardised.geojson").json();
const parcelData = FileAttachment("./data/venice_1808_landregister_geometries.geojson").json();
const registre = FileAttachment("./data/venice_1808_landregister_textual_entries.json").json();
```

<!-- Create the map container -->
<div id="map-container-casa-average-size-hm" class="map-component"></div>

```js
// Call the creation function and store the results
const porzioneMapComponents = createParishCasaAverageSurfaceHeatMap("map-container-casa-average-size-hm", parcelData, registre, parishData);

// affecting values to the window is the easiest way to break the observable sandbox and make code available in the plain JS context of the webpage.
window.highlightFeature = (name) => {
    porzioneMapComponents.geoJsonLayerAverage.resetStyle();
    // for some reason, observable does not let me set intermediat variable, so all action on layer has to call the layer from the hashMap again.
    porzioneMapComponents.map.flyTo(porzioneMapComponents.parishNameLayerMap.get(name).getBounds().getCenter(), 15.4);
    porzioneMapComponents.parishNameLayerMap.get(name).setStyle({
        weight: 5,
        color: '#FFF',
        dashArray: '',
        fillOpacity: 0.7
    });
    porzioneMapComponents.parishNameLayerMap.get(name).bringToFront();
    porzioneMapComponents.parishNameLayerMap.get(name).openPopup() 
    // document.getElementById('map-container-casa-average-size-hm').scrollIntoView({"behavior":"smooth"});
};
```

<!-- Create the tanble container -->
<div class="block-container">
<div id="table-container-casa-surface-ranking"></div>
</div>

```js
const table = Inputs.table(porzioneMapComponents.tableData, {
    header: {
        name: "Parish Name",
        average_surface: "Average parcel area (m2)",
        median_surface: "Median parcel area (m2)"
    },
    format: {
        name: id => htl.html`<a class="hover-line table-row-padding" onclick=window.highlightFeature("${id}");>${id}</a>`,
       average_surface: x => x.toFixed(1),
       median_surface: x => x.toFixed(1),
    }, 
    select: false
});
document.getElementById("table-container-casa-surface-ranking").append(table)
```

This map visualizes the **average and median parcel surface area by parish**, calculated exclusively from parcels where the **term “house”** appears in the cadastral “Quality” field. This filtering ensures that the analysis focuses specifically on **residential properties**, excluding warehouses, workshops, religious buildings, and other non-domestic uses. The results highlight notable contrasts across the city.

An analysis of **median parcel area**, excluding extreme high and low values, highlights several parishes where the **residential fabric is notably dense**, composed of **many small housing plots**. These **Compact urban zones** include the **Ghetto districts** (Ghetto Nuovo, Vecchio, and Nuovissimo), the **Realtina area**, and the **central parishes** of **San Luca** and **San Zeminian**. Another relevant example is the **western offshoot of San Nicolò** (now known as Santa Marta), which also exhibits this pattern. These parishes likely reflect **working-class or modest residential zones**, where land subdivision and space optimization resulted in smaller, tightly packed parcels.

In contrast, **parishes with a smaller number of larger residential plots**—such as **San Marco or Santa Croce**—reveal a different urban dynamic. These areas likely contain **palatial or patrician dwellings**, or large properties with extensive open spaces, gardens, or courtyards, resulting in higher median and average surface values.

It is important to note potential **statistical biases**, especially in **small parishes** where a single large estate may skew the results. For example, in **San Boldo**, the presence of Palazzo Civran, a substantial housing parcel, disproportionately raises the average surface area per parcel, despite the limited number of residential units in the parish.