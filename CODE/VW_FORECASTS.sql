SELECT 
    "FACT_SAVED_FORECASTS"."Forecast_as_at_date"::DATE,
    "FACT_SAVED_FORECASTS"."Horizon"::INTEGER,
    left(right("FACT_SAVED_FORECASTS"."Predicted - Descaled",-1),-1)::FLOAT as "Forecasts"
FROM "FACT_SAVED_FORECASTS"

ORDER BY
    "FACT_SAVED_FORECASTS"."Forecast_as_at_date"::DATE,
    "FACT_SAVED_FORECASTS"."Horizon"::INTEGER