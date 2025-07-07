// See https://observablehq.com/framework/config for documentation.
export default {
  // The app’s title; used in the sidebar and webpage titles.
  title: "Napoleonic Cadaster - Thematic Maps",

  // The pages and sections in the sidebar. If you don’t specify this option,
  // all pages will be listed in alphabetical order. Listing pages explicitly
  // lets you organize them into sections and have unlisted pages.
  pages: [
    {
      name: "Thematic Vizualisations",
      pages: [
        {name: "\"Portion Heatmap\" - Fragmentation of Domestic Space in 1808 Venice", path: "/cadastre-porzione-heatmap"},
        {name: "Parcel's functions and standardised classes", path: "/cadastre-property-type"},
        {name: "Average surface of “house” function per parish", path: "/parish-average-casa-size"},
        {name: "Expropriations of private properties", path: "/expropriation-map"},
        {name: "Functions of Expropriated Parcels by Receiving Institution", path: "/function-histograms"},
        {name: "Type of Geometry", path: "/geometry-type-maps"},
        {name: "Chronology Test", path: "/chronology_test"}
      ]
    }
  ],

  // Content to add to the head of the page, e.g. for a favicon:
  head: '<link rel="icon" href="observable.png" type="image/png" sizes="32x32">',

  // The path to the source root.
  root: "src",

  // Some additional configuration options and their defaults:
  theme: "light", // try "light", "dark", "slate", etc.
  // header: "", // what to show in the header (HTML)
  footer: "©Time Machine Unit | EPFL", // what to show in the footer (HTML)
  // sidebar: true, // whether to show the sidebar
  // toc: true, // whether to show the table of contents
  pager: false, // whether to show previous & next links in the footer
  // output: "dist", // path to the output root for build
  // search: true, // activate search
  // linkify: true, // convert URLs in Markdown to links
  // typographer: false, // smart quotes and other typographic improvements
  // preserveExtension: false, // drop .html from URLs
  // preserveIndex: false, // drop /index from URLs
};
