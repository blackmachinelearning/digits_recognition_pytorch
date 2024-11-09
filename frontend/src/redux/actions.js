// redux/actions.js

export const setProbability = (probabilities) => ({
    type: "SET_PROBABILITY",
    payload: probabilities,
  });
  
  export const clearProbability = () => ({
    type: "CLEAR_PROBABILITY",
  });
  