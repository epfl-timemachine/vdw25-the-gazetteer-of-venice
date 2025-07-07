---
title: test chronology
toc: false
style: components/custom-style.css
---


Test for the chronology thing

```js
const parcelData = FileAttachment("./data/place_entity_df.csv").csv({typed:true});

console.log(parcelData)
```
```js
const searchResults = view(Inputs.search(parcelData))
console.log(searchResults)
```

<!-- ```js
const searchResults = []
```

```js
const test = view(
  Inputs.select(parcelData, {
    label: "Places",
    format: (t) => t.Place,
  })
);

const add_button = view(Inputs.button("Add Place"))

```

```js
add_button; // run this block when the button is clicked
const progress = (function* () {
  searchResults.push(test)
  console.log(searchResults)
})();
``` -->



```js
//add_button;
Plot.plot({
  width: 1000,
  height: 700,
  marginLeft: 400,
  grid: true,
  marks: [
    Plot.dot(searchResults, Plot.pointer({x: "Year", y: "Place", tip: "x"})),
    Plot.dot(searchResults, {x: "Year", y: "Place", title: d => `Place: ${d.Place}\nYear: ${d.Year}}`})
  ],
})
```

