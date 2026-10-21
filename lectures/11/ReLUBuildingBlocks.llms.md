# Building Functions with ReLU

Code

- [Show All Code](javascript:void(0))

- [Hide All Code](javascript:void(0))

- 

  ------------------------------------------------------------------------

- [View Source](javascript:void(0))

An interactive companion to Lecture 11

Author

Marcel Turcotte

Published

Version: Oct 20, 2026

Change the seven parameters of a network with **one input, two hidden ReLU units, and one linear output**. Start by changing one parameter at a time.

``` math
f(x)=w_3\varphi(w_1x+b_1)+w_4\varphi(w_2x+b_2)+b_3.
```

Here $`\varphi(z)=\max(0,z)`$ is ReLU. The coloured curves show the two **hidden activations**. The thicker curve combines them using the output weights $`w_3`$ and $`w_4`$, then adds the output bias $`b_3`$.

Show the Observable code

``` js
networkDiagram = {
  const nodes = {
    input: {x: 100, y: 260, r: 36, type: "input", fill: "#afffff"},
    inputBias: {x: 100, y: 65, r: 24, type: "constant", fill: "#eeeeee"},
    hidden1: {x: 470, y: 170, r: 44, type: "relu", fill: "#bfbfff", accent: "#c26817", caption: "Unit 1"},
    hidden2: {x: 470, y: 350, r: 44, type: "relu", fill: "#bfbfff", accent: "#3679b0", caption: "Unit 2"},
    hiddenBias: {x: 470, y: 65, r: 24, type: "constant", fill: "#eeeeee"},
    output: {x: 850, y: 260, r: 44, type: "linear", fill: "#80ff80", caption: "f(x)"}
  };
  const edges = [
    {from: "input", to: "hidden1", name: "w1", symbol: "w₁", lx: 285, ly: 206},
    {from: "input", to: "hidden2", name: "w2", symbol: "w₂", lx: 285, ly: 323},
    {from: "hidden1", to: "output", name: "w3", symbol: "w₃", lx: 650, ly: 190},
    {from: "hidden2", to: "output", name: "w4", symbol: "w₄", lx: 650, ly: 325},
    {from: "inputBias", to: "hidden1", name: "b1", symbol: "b₁", lx: 275, ly: 102},
    {from: "inputBias", to: "hidden2", name: "b2", symbol: "b₂", lx: 225, ly: 170},
    {from: "hiddenBias", to: "output", name: "b3", symbol: "b₃", lx: 680, ly: 155}
  ];
  const formatValue = value => value.toLocaleString("en-CA", {maximumFractionDigits: 1});

  const drawEdge = edge => {
    const start = nodes[edge.from], end = nodes[edge.to];
    const dx = end.x - start.x, dy = end.y - start.y;
    const length = Math.hypot(dx, dy);
    return svg`<line data-parameter=${edge.name} data-from=${edge.from} data-to=${edge.to}
      x1=${start.x + dx * start.r / length} y1=${start.y + dy * start.r / length}
      x2=${end.x - dx * (end.r + 3) / length} y2=${end.y - dy * (end.r + 3) / length}
      stroke="currentColor" stroke-width="2" marker-end="url(#relu-network-arrow)"></line>`;
  };

  const drawNode = ([name, node]) => svg`<g data-node=${name} transform="translate(${node.x},${node.y})">
    <circle r=${node.r} fill=${node.fill} stroke=${node.accent || "#111"} stroke-width="2.5"></circle>
    ${node.type === "input" || node.type === "constant"
      ? svg`<text text-anchor="middle" y="10" font-size=${node.type === "input" ? 36 : 28}
          font-style=${node.type === "input" ? "italic" : "normal"} fill="#111">${node.type === "input" ? "x" : "1"}</text>`
      : svg`<g fill="#111" stroke="#111">
          <line y1=${-node.r} y2=${node.r} stroke-width="2"></line>
          <text x="-20" y="12" text-anchor="middle" font-size="39" stroke="none">∑</text>
          <path d="${node.type === "relu" ? "M 6 15 H 16 L 34 -15" : "M 7 17 L 34 -17"}"
            fill="none" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></path>
        </g>`}
    ${node.caption ? svg`<text text-anchor="middle" y=${node.r + 25}
      font-family="${node.type === "linear" ? "Georgia, serif" : "system-ui, sans-serif"}"
      font-size="18" fill=${node.accent || "currentColor"}>${node.caption}</text>` : null}
  </g>`;

  return html`<figure class="relu-network-figure">
    <div class="relu-network-scroll">
      ${svg`<svg class="relu-network-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 440"
        role="img" aria-labelledby="relu-network-title relu-network-description" font-family="Georgia, serif">
        <title id="relu-network-title">One input, two hidden ReLU units, and one linear output</title>
        <desc id="relu-network-description">The input x connects to both hidden units through w₁ and w₂.
          The hidden units connect to the output through w₃ and w₄.
          A constant node above x supplies biases b₁ and b₂; a second constant node supplies b₃ to the output.
          Neither constant node has incoming edges.</desc>
        <defs>
          <marker id="relu-network-arrow" viewBox="0 -4 8 8" refX="8" refY="0"
            markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">
            <path d="M 0 -4 L 8 0 L 0 4 Z" fill="currentColor"></path>
          </marker>
        </defs>
        <g text-anchor="middle" font-family="system-ui, sans-serif" font-size="17" fill="currentColor">
          <text x="100" y="20">Input</text>
          <text x="470" y="20">Hidden layer (ReLU)</text>
          <text x="850" y="20">Linear output</text>
        </g>
        ${edges.map(drawEdge)}
        ${Object.entries(nodes).map(drawNode)}
        ${edges.map(edge => svg`<text class="parameter-label" data-label=${edge.name}
          x=${edge.lx} y=${edge.ly} text-anchor="middle">${edge.symbol} = ${formatValue(parameters[edge.name])}</text>`)}
      </svg>`}
    </div>
    <figcaption>The small nodes always emit 1 and have no incoming edges. Their outgoing weights are the biases.
      Parameter labels update with the sliders; the orange and blue outlines identify the two hidden activations.</figcaption>
  </figure>`;
}
```

Show the Observable code

``` js
viewof parameters = {
  const initial = {w1: 2, b1: 3, w3: 1, w2: 2, b2: 0, w4: -2, b3: 3};
  const labels = {w1: "w₁", b1: "b₁", w3: "w₃", w2: "w₂", b2: "b₂", w4: "w₄", b3: "b₃"};
  const inputs = Object.fromEntries(
    Object.entries(initial).map(([name, value]) => {
      const input = Inputs.range([-3, 3], {label: labels[name], value, step: 0.1, width: 190});
      input.querySelector('input[type="range"]').setAttribute("aria-label", labels[name]);
      return [name, input];
    })
  );

  const form = Inputs.form(inputs, {
    template: fields => html`<div class="relu-controls">
      <fieldset style="--unit-colour: #c26817">
        <legend>Hidden unit 1</legend>
        <p class="unit-caption">Input weight, bias, output weight</p>
        ${fields.w1}${fields.b1}${fields.w3}
      </fieldset>
      <fieldset style="--unit-colour: #3679b0">
        <legend>Hidden unit 2</legend>
        <p class="unit-caption">Input weight, bias, output weight</p>
        ${fields.w2}${fields.b2}${fields.w4}
      </fieldset>
      <fieldset style="--unit-colour: currentColor">
        <legend>Output</legend>
        <p class="unit-caption">Vertical shift of the complete curve</p>
        ${fields.b3}
      </fieldset>
      <div><button type="button" class="btn btn-outline-secondary btn-sm">Reset parameters</button></div>
    </div>`
  });

  form.querySelector("button").addEventListener("click", () => {
    form.value = initial;
    form.dispatchEvent(new Event("input", {bubbles: true}));
  });

  return form;
}
```

Show the Observable code

``` js
viewof yExtent = Inputs.select([8, 16, 40, 80], {
  value: 8,
  label: "Vertical range",
  format: extent => `−${extent} to +${extent}`
})
```

Show the Observable code

``` js
Plot.plot({
  width: Math.min(width, 850),
  height: 410,
  marginLeft: 50,
  marginRight: 20,
  x: {domain: [-3, 3], label: "x", grid: true},
  y: {domain: [-yExtent, yExtent], label: "Output", grid: true},
  color: {
    domain: ["Unit 1 activation", "Unit 2 activation", "Network output"],
    range: ["#c26817", "#3679b0", "currentColor"],
    legend: true
  },
  marks: [
    Plot.ruleY([0], {strokeOpacity: 0.35}),
    Plot.ruleX(breakpoints, {
      x: "x", stroke: "series", strokeDasharray: "4,4", strokeOpacity: 0.45,
      clip: true
    }),
    Plot.lineY(samples, {
      x: "x", y: "activation1", stroke: () => "Unit 1 activation",
      strokeWidth: 2, clip: true
    }),
    Plot.lineY(samples, {
      x: "x", y: "activation2", stroke: () => "Unit 2 activation",
      strokeWidth: 2, clip: true
    }),
    Plot.lineY(samples, {
      x: "x", y: "output", stroke: () => "Network output",
      strokeWidth: 3.5, clip: true
    })
  ]
})
```

The axes stay fixed when parameters change. Use the vertical-range selector if a curve extends outside the current view. Dashed vertical lines mark the hidden units’ breakpoints.

Show the Observable code

``` js
html`<div class="relu-breakpoints" aria-live="polite">
  <p><strong>Unit 1:</strong> ${describeBreakpoint(parameters.w1, parameters.b1)}</p>
  <p><strong>Unit 2:</strong> ${describeBreakpoint(parameters.w2, parameters.b2)}</p>
</div>`
```

## From two units to more

The output curve combines the two hidden activations. Its slope can change wherever either activation changes slope. The output weights $`w_3`$ and $`w_4`$ scale the activations and can make their contributions reinforce or oppose each other, while $`b_3`$ moves the entire result vertically.

Imagine adding more hidden units. Where could you place new bends, and how could their output weights help shape the segments between them? Combining more units gives us more building blocks for constructing complex functions, connecting this exploration to the lecture’s discussion of approximation with wider hidden layers.

Try setting $`w_3=0`$: the first hidden activation stays visible, but no longer contributes to the network output. Changing the sign of $`w_4`$ reverses the second unit’s contribution to the output while leaving its blue activation curve unchanged. Use **Reset parameters** to restore the starting network.

To experiment with more hidden units, try the [Neural Network Visualizer](https://kickereb.github.io/nn-visualizer/experiment-2d.html). Add units and observe how the network’s regression curve changes. Its hidden units use $`\tanh`$ activations.

[Return to Lecture 11](slides.llms.md#universal-approximation).

## How the curves are computed

The browser evaluates the network at points across $`[-3,3]`$. Breakpoints inside the interval are included explicitly so the plotted line passes through each bend.

Show the Observable code

``` js
relu = z => Math.max(0, z)

breakpoints = [
  {w: parameters.w1, b: parameters.b1, series: "Unit 1 activation"},
  {w: parameters.w2, b: parameters.b2, series: "Unit 2 activation"}
].filter(d => d.w !== 0)
  .map(d => ({x: -d.b / d.w, series: d.series}))
  .filter(d => d.x >= -3 && d.x <= 3)

samples = {
  const xs = Array.from({length: 601}, (_, i) => -3 + i / 100);
  xs.push(...breakpoints.map(d => d.x));
  return [...new Set(xs)].sort((a, b) => a - b).map(x => {
    const activation1 = relu(parameters.w1 * x + parameters.b1);
    const activation2 = relu(parameters.w2 * x + parameters.b2);
    const output = parameters.w3 * activation1 + parameters.w4 * activation2 + parameters.b3;
    return {x, activation1, activation2, output};
  });
}

describeBreakpoint = (w, b) => {
  if (w === 0) return "Constant activation; no breakpoint.";
  const x = -b / w;
  return `x = ${x.toFixed(2)}${x < -3 || x > 3 ? " (outside the plotted interval)" : ""}`;
}
```
