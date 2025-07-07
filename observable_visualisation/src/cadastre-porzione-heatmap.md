---
title: Napoleonic Cadaster - Porzione Heatmap
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
import {createPorzioneHeatMap} from "./components/map-porzione.js";
```

# "Portion Heatmap" - Fragmentation of Domestic Space in 1808 Venice.

```js
const geojson = FileAttachment("./data/venice_1808_landregister_geometries.geojson").json();
const registre = FileAttachment("./data/venice_1808_landregister_textual_entries.json").json();
```

<!-- Create the map container -->
<div id="map-container-porzione-hm" class="map-component"></div>

```js
// Call the creation function and store the results
const porzioneMapComponents = createPorzioneHeatMap("map-container-porzione-hm", geojson, registre);
```
This thematic map visualizes the frequency of the word “**portion**” in the cadastral dataset, as it appears in the “**Quality**” field alongside urban functions. The most common phrase is “**portion of house**”, pointing to a striking phenomenon in the structure of the built environment: the **internal fragmentation of housing units**.

These "portions" typically refer to parts of a single building—often individual floors—owned by different individuals. This suggests that buildings originally constructed as unified homes were later **subdivided into multiple residential units**, likely to accommodate several households within a constrained urban fabric.

The pattern becomes even more significant when considering the number of distinct owners per parcel. In some cases, the heatmap reveals **eight or more owners** for properties that consist of only **two or three floors**. This degree of subdivision indicates a form of **micro-property ownership**, possibly reflecting economic strategies to enable families of modest means to access urban housing through partial acquisition or rental arrangements.

The map thus highlights not only a spatial pattern, but also a **social and economic adaptation**: a city responding to demographic pressure and evolving needs by reconfiguring its interior domestic space—layer by layer, owner by owner.
