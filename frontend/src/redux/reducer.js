const initialState = {
  probabilities: [],
};

export default function reducer(state = initialState, action) {
  switch (action.type) {
    case "SET_PROBABILITY":
      return {
        ...state,
        probabilities: action.payload,
      };
    default:
      return state;
  }
}
