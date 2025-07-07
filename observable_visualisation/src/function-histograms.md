---
title: Napoleonic Cadaster - Functions Histograms
toc: false
style: components/custom-style.css
---


# Functions of Expropriated Parcels by Receiving Institution


<!-- Create the tanble container -->
<div class="block-container">
<div id="barchart-container" class="block-component"></div>
<div id="barchart-legend"></div>
</div>

```js
import {cookData, cookDataInSurfaceArea} from "./components/function-data-cooking.js";
const registre = FileAttachment("./data/venice_1808_landregister_textual_entries.json").json();
const parcelData = FileAttachment("./data/venice_1808_landregister_geometries.geojson").json();
```

```js
const plotWidth = 1200;
const plotHeight = 750;
const marginLeft = 50;
const cookedData = cookData(registre, 10);
const sBarChart = Plot.plot({
  width: plotWidth,
  height: plotHeight,
  y: {tickFormat: "s", tickSpacing: 50},
  marginLeft: marginLeft,
  color: {
    scheme: "Spectral",
    type: "categorical", 
    columns: 3,
    legend: true,
    width: plotWidth - 400,
    marginLeft: marginLeft
  },
  marks: [
    Plot.barY(cookedData, {
        x: "name",
        y: "count",
        fill: "quality",
        title: v => `${v.quality}: ${v.count}`,
        sort: {x: "-y"},
        tip: true
      }
    ),
    Plot.axisX({label: null, lineWidth: 8, marginBottom: 40}),
  ],
  tooltip: {
    fill: "white",
    stroke: "blue",
    r: 8
  }
});

document.getElementById('barchart-container').append(sBarChart)
```
The histogram displays the **absolute count of parcel functions** for the ten most significant standardized owners in the Napoleonic cadaster. These are the institutions or entities that acquired the highest number of parcels following the expropriation of ecclesiastical and charitable property. Each bar is broken down by function (e.g., _house, shop, garden, church, warehouse,_ etc.), revealing both the **quantitative and functional composition** of each entity’s newly acquired holdings.

Several key patterns emerge:
* **The "Città di Venezia"** received the greatest number and most functionally diverse set of parcels, with strong representation of _house, shop, garden_, and _warehouse_ functions—suggesting an absorption of a wide swath of urban functions, possibly as municipal infrastructure or redistributed housing stock. 
* The "**Comune di Venezia**" follows closely, but with a markedly higher proportion of *“Scuole”*  functions. 
* **The "Congregazione di Carità"**, a secularized continuation of older charitable institutions, also presents a balanced mix, indicating its role in managing former ecclesiastical assets for social use.
* Ministries like the **Ministero della Guerra** and **Ministero delle Finanze** display functions consistent with **institutional or military reuse**, such as *barracks* or *storehouses*.
* Former private or ecclesiastical owners such as **Chiesa di San Pantaleon** or **Mocenigo Alvise** show residual ownership, sometimes including church, monastery, or courtyard functions—likely parcels that remained under their nominal control before full redistribution.

This visualization underscores the **functional fragmentation and reallocation of the urban fabric** triggered by Napoleonic reforms. It also illustrates the **institutional transformation of property** in post-republican Venice: from sacred to civic, from monastic to military, from religious to residential.


<!-- Create the tanble container -->
<div class="block-container">
<div id="barchart-surface-container" class="block-component"></div>
<div id="barchart-surface-legend"></div>
</div>

```js
const cookedDataSurface = cookDataInSurfaceArea(registre, parcelData, 10);
const sBarChartSurface = Plot.plot({
  width: plotWidth,
  height: plotHeight,
  y: {tickFormat: "s", tickSpacing: 50},
  marginLeft: marginLeft,
  color: {
    scheme: "Spectral",
    type: "categorical", 
    columns: 3,
    legend: true,
    width: plotWidth - 400,
    marginLeft: marginLeft
  },
  marks: [
    Plot.barY(cookedDataSurface, {
        x: "name",
        y: "surface",
        fill: "quality",
        title: v => `${v.quality}: ${v.surface.toFixed(1)}m2`,
        sort: {x: "-y"},
        tip: true
      }
    ),
    Plot.axisX({label: null, lineWidth: 8, marginBottom: 40}),
  ],
  tooltip: {
    fill: "white",
    stroke: "blue",
    r: 8
  }
});

document.getElementById('barchart-surface-container').append(sBarChartSurface);
```

This histogram presents the **total surface area (in square meters)** attributed to different parcel functions for the ten most significant standardized property holders following the Napoleonic expropriations. Unlike the previous histogram based on **parcel counts**, this visualization measures **land surface**, thereby highlighting ownership of **large estates, open areas, and non-residential parcels**.

Key insights include:
* **Città di Venezia** again dominates in total surface, with its holdings covering a vast range of functions. The most extensive functions include *house*, *courtyard*, and *warehouse*, along with significant surface allocated to *uncultivated land* and *gardens*. This indicates a mixed portfolio of residential and infrastructural use, often across formerly ecclesiastical lands.
* The **Ministero della Guerra** appears as a key receiver of large properties, notably with _barracks_, _arsenale_, and _open areas_, suggesting conversion of expropriated land for military and logistical purposes.
* **Comune di Venezia** continues to reflect a strong civic use of acquired land, including *“Scuole”*, *kitchen gardens*, and *spaces*.
* **Congregazione di Carità**, in line with its mission, displays surface holdings devoted to *household*, *garden*, and *hospital* functions—highlighting its role in redistributing welfare functions previously held by religious orders.
* Smaller yet still notable landowners like **Ministero delle Finanze** or **Regio Demanio** manage diverse properties with moderate emphasis on *warehouse*, *house*, and *space*.

This histogram complements the absolute count analysis by showing **how land function translates into spatial dominance**. Some functions (e.g. *yards*, *uncultivated gardens*, *courtyards*) occupy large areas but are underrepresented in count, while others (e.g. *shop*, *hallway*) are numerous but occupy limited space. The histogram thus highlights the **material footprint of institutional change**. 
