---
toc: false
---

<div class="hero">
  <h1>Napoleonic Cadaster - Thematic Maps</h1>
  <!-- <h2>Welcome to your new app! Edit&nbsp;<code style="font-size: 90%;">src/index.md</code> to change this page.</h2>
  <a href="https://observablehq.com/framework/getting-started">Get started<span style="display: inline-block; margin-left: 0.25rem;">↗︎</span></a> -->
</div>
<!-- 
<div class="grid grid-cols-2" style="grid-auto-rows: 504px;">
  <div class="card">${
    resize((width) => Plot.plot({
      title: "Your awesomeness over time 🚀",
      subtitle: "Up and to the right!",
      width,
      y: {grid: true, label: "Awesomeness"},
      marks: [
        Plot.ruleY([0]),
        Plot.lineY(aapl, {x: "Date", y: "Close", tip: true})
      ]
    }))
  }</div>
  <div class="card">${
    resize((width) => Plot.plot({
      title: "How big are penguins, anyway? 🐧",
      width,
      grid: true,
      x: {label: "Body mass (g)"},
      y: {label: "Flipper length (mm)"},
      color: {legend: true},
      marks: [
        Plot.linearRegressionY(penguins, {x: "body_mass_g", y: "flipper_length_mm", stroke: "species"}),
        Plot.dot(penguins, {x: "body_mass_g", y: "flipper_length_mm", stroke: "species", tip: true})
      ]
    }))
  }</div>
</div> -->

---

## Interactive data explorer
The 1808 Napoleonic cadastre of Venice is more than an administrative artifact—it is a cartographic lens on a city in transition, revealing the legal, social, and spatial frameworks that governed its urban structure at the dawn of the 19th century. Born out of a radical reorganization of public authority under Napoleonic rule, the cadastre recorded for the first time, with geometric precision, the entirety of Venice’s built environment, its parcels, owners, and the declared uses of its properties. Now digitized, semantically enriched, and made available through an interactive platform—**TimeAtlas**—this material forms a new kind of urban archive: one that can be queried, visualized, and extended collaboratively.

Unlike traditional historical records of surveys, which often privilege narrative over structure, the geometric cadastre is explicitly spatial, legal, and economic in nature. It does not merely describe who lived where, but how parcels were fragmented, owned, subdivided, and used. Through the combined work of data transcription, normalization, and semantic annotation, the cadastral dataset has been transformed into a dynamic environment for historical exploration. Thousands of parcel entries have been processed to detect functional mentions—residences, shops, warehouses, gardens, or civic uses—and to trace titles, and ownership structures. Each element of the data is now spatially linked and searchable, allowing for large-scale comparison as well as micro-historical analysis.

This open structure—combining a research paper, a GitHub repository, and a public web interface—offers a methodological shift in how historical theories are produced and communicated. Rather than relying solely on argumentation, the project anchors interpretation in an openly accessible and inspectable data environment. Every map and visualization can be retraced to the underlying data, which is both documented and transparent. This makes the project verifiable, extensible, and reusable by a broad range of users: historians, urban planners, digital humanists, and even local communities.

A crucial insight from this work is the extreme granularity of the Venetian housing fabric. One striking indicator is the prevalence of the term “_portion_” in parcel descriptions, as in “_portion of house_”. These refer not to rooms but to property units owned by different individuals—often up to eight in the same plot—revealing a regime of ownership fragmentation that responded to the dense population and intense pressure on housing. This kind of mention captures how vertical divisions of property (by floor or section) became a strategy for accommodating multiple households and distributing access to real estate in a city where space was a limited and contested resource.
The cadastre also allows us to examine not just what buildings were, but how they were used. Mentions in the “Quality” field—standardized across the dataset—distinguish properties occupied by their owners from those leased to others. In a city already deeply shaped by a rental economy, the interface reveals a landscape dominated by leased property, with only limited portions reserved for owner use. Other functions, such as warehouses, shops, and gardens, cluster in recognizable urban morphologies that can now be visualized and analyzed with precision.

What makes these insights particularly meaningful is their entanglement with ownership structures. The dataset encodes the aftermath of a major transformation in Venice’s institutional property regime: the suppression and expropriation of religious and charitable institutions following the fall of the Republic. By comparing "old" and "current" ownership classes, users can see how parcels once held by monasteries, confraternities, and the Scuole Grandi passed into public hands. Some were given to the newly formed ministries, others to military or financial offices, and many to the municipal government. These transitions are more than bureaucratic: they redraw the ideological map of the city. Institutions that had structured daily life for centuries disappeared almost overnight from the property register, replaced by impersonal public ownership. Yet their material legacy remains visible in the built environment and in the metadata of the cadastral archive.

A key benefit of this interface is its capacity to expose such dynamics not only through historical narrative but through interactive exploration. Users can visualize the distribution of property types, the average or median size of housing parcels across parishes, or the rights of use claimed by religious institutions. They can compare distributions of owner categories, trace patterns of expropriation, and see where institutional power was geographically concentrated. In doing so, the research user does not merely consume conclusions; they test hypotheses, verify assertions, and build their own interpretations from the same dataset.

Moreover, the granularity of the dataset allows for new thematic research. For example, the project has identified how ecclesiastical owners specified rights of use in detail—benefices, prebends, livellary rights—offering a previously unseen picture of how clerical property was managed. Even the terminology used to describe institutions (_monastery vs. convent_) appears flexible and ambiguous, challenging assumptions about fixed religious typologies. Statistical anomalies—such as the outsized impact of individual palaces on average parcel size in small parishes—highlight the need to contextualize data interpretation historically.

Finally, the digitized map itself becomes a powerful interpretive layer. The original color codes used by the 1808 cadastral office—pink for buildings, blue for water, beige for courtyards—have been preserved and embedded in the digital geometries. This not only creates a faithful “digital twin” of the map but also reinforces the logic of spatial legibility that underpinned the original document. The cadastral map was not just a survey; it was a tool of governance, taxation, and legal control. Now, it becomes a platform for historical imagination.

The maps reveal the conditions under which Venice’s urban system was redefined, not only by new political regimes but also by the practices of people: co-owning a floor, renting a portion, repurposing a convent, subdividing a home. The cadastre is where legal form, spatial structure, and social negotiation converge. 


<!-- 
Here are some ideas of things you could try…

<div class="grid grid-cols-4">
  <div class="card">
    Chart your own data using <a href="https://observablehq.com/framework/lib/plot"><code>Plot</code></a> and <a href="https://observablehq.com/framework/files"><code>FileAttachment</code></a>. Make it responsive using <a href="https://observablehq.com/framework/javascript#resize(render)"><code>resize</code></a>.
  </div>
  <div class="card">
    Create a <a href="https://observablehq.com/framework/project-structure">new page</a> by adding a Markdown file (<code>whatever.md</code>) to the <code>src</code> folder.
  </div>
  <div class="card">
    Add a drop-down menu using <a href="https://observablehq.com/framework/inputs/select"><code>Inputs.select</code></a> and use it to filter the data shown in a chart.
  </div>
  <div class="card">
    Write a <a href="https://observablehq.com/framework/loaders">data loader</a> that queries a local database or API, generating a data snapshot on build.
  </div>
  <div class="card">
    Import a <a href="https://observablehq.com/framework/imports">recommended library</a> from npm, such as <a href="https://observablehq.com/framework/lib/leaflet">Leaflet</a>, <a href="https://observablehq.com/framework/lib/dot">GraphViz</a>, <a href="https://observablehq.com/framework/lib/tex">TeX</a>, or <a href="https://observablehq.com/framework/lib/duckdb">DuckDB</a>.
  </div>
  <div class="card">
    Ask for help, or share your work or ideas, on our <a href="https://github.com/observablehq/framework/discussions">GitHub discussions</a>.
  </div>
  <div class="card">
    Visit <a href="https://github.com/observablehq/framework">Framework on GitHub</a> and give us a star. Or file an issue if you’ve found a bug!
  </div>
</div> -->

<style>

.hero {
  display: flex;
  flex-direction: column;
  align-items: center;
  font-family: var(--sans-serif);
  margin: 4rem 0 8rem;
  text-wrap: balance;
  text-align: center;
}

.hero h1 {
  margin: 1rem 0;
  padding: 1rem 0;
  max-width: none;
  font-size: 14vw;
  font-weight: 900;
  line-height: 1;
  background: linear-gradient(30deg, var(--theme-foreground-focus), currentColor);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero h2 {
  margin: 0;
  max-width: 34em;
  font-size: 20px;
  font-style: initial;
  font-weight: 500;
  line-height: 1.5;
  color: var(--theme-foreground-muted);
}

@media (min-width: 640px) {
  .hero h1 {
    font-size: 90px;
  }
}

</style>
