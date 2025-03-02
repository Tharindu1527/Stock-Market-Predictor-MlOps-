import { useCallback } from "react";
import axios from "axios";
import useData from "../globalVariables/dataContext";

// Use environment variables or determine the base URL dynamically
const API_URL = import.meta.env.PROD
  ? "https://stock-predictions-backend-production.up.railway.app/api"
  : "/api";

const client = axios.create({
  baseURL: API_URL
});

export default function useRequestResource() {
  const {
    getHistoryPrices, getTimeSeries, getTrainTime,
    getTrainPrices, getPriceClose, getPredictionClose,
    getTimePrediction, getRmse, Loading, getPredPrice
  } = useData();

  const getResourceData = useCallback(
    ({ query }: { query: string }) => {
      Loading();
      
      client.get(`${query}`)
        .then((res) => {
          console.log("API Response:", res.data);
          // History Prices
          getHistoryPrices(res.data.data.prices);
          getTimeSeries(res.data.data.time);
          // Train Data
          getTrainTime(res.data.data.train.timeTrain);
          getTrainPrices(res.data.data.train.Close);
          // Table Data
          getPriceClose(res.data.data.valid.Close);
          getPredictionClose(res.data.data.valid.Predictions);
          getTimePrediction(res.data.data.valid.timeValid);
          // Rmse
          getRmse([res.data.data.rmse]);
          // prediction price - this is the key fix
          getPredPrice([res.data.data.predicted_price]);
          Loading();
        })
        .catch((err) => {
          console.error("API Error:", err);
          Loading(); // Make sure to reset loading state on error
        });
    },
    [getHistoryPrices, getTimeSeries, getTrainTime, getTrainPrices, 
     getPriceClose, getPredictionClose, getTimePrediction, getRmse, 
     Loading, getPredPrice]
  );

  return {
    getResourceData,
  };
}