// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.


import * as React from "react";
import { createRoot } from "react-dom/client";
import AppContextProvider from "./components/hooks/context";
import App from "./App";
const container = document.getElementById("root");
const root = createRoot(container!);
root.render(
  <AppContextProvider>
    <App/>
  </AppContextProvider>
);
