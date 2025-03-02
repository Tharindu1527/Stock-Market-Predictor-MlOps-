import React from "react";
import useData from "../globalVariables/dataContext";
import { Typography, Box, CircularProgress } from "@mui/material";

export default function Prediction() {
  const { price, loading } = useData();
  
  // Log the current state for debugging
  console.log("Prediction component - price:", price, "loading:", loading);
  
  // Safe access to the price data
  const predprice = (!loading && price && price.length > 0)
    ? Number(price[0]).toFixed(2)
    : null;
  
  return (
    <Box>
      <Typography
        variant="h3"
        sx={{
          pt: 5,
          pb: 4,
          fontFamily: "Roboto Flex",
          color: "rgba(255,255,255,0.8)",
          fontSize: { lg: 45, md: 45, sm: 35, xs: 25 },
        }}
      >
        Prediction
      </Typography>
      <Box
        sx={{
          bgcolor: loading || !predprice ? "rgba(20,20,40,0.8)" : "rgba(3,255,249,0.8)",
          width: { lg: "200px", md: "200px", sm: "150px" },
          borderRadius: "10px",
          p: 2,
          textAlign: "center",
          fontSize: { lg: 25, md: 25, sm: 18, xs: 17 },
          boxShadow: loading || !predprice ? "none" : "0px 0px 15px #03FFF9",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        {loading ? (
          <CircularProgress size={30} color="info" />
        ) : predprice ? (
          `$${predprice}`
        ) : (
          "Select a stock"
        )}
      </Box>
    </Box>
  );
}