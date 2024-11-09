// redux/store.js

import { configureStore } from "@reduxjs/toolkit";
import probabilityReducer from "./reducer";

const store = configureStore({
  reducer: {
    probability: probabilityReducer, // You can add more reducers here if needed
  },
});

export default store;
