// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.


const tailwindcss = require("tailwindcss");
module.exports = {
  plugins: ["postcss-preset-env", 'tailwindcss/nesting', tailwindcss],
};
