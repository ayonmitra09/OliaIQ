from src.valuation_adjustment import apply_demographics_adjustment
from src.address_mapper import provide_fips_codes

# Get user input for address and valuation
address = input("Please enter the property address: ")
try:
    valuation = float(input("Please enter the current property valuation: $"))
except ValueError:
    print("Invalid valuation amount. Please enter a numeric value.")
    exit(1)

# Get FIPS codes from address
try:
    state_code, county_code = provide_fips_codes(address)
except ValueError as e:
    print(f"Error: {e}")
    exit(1)

# Apply demographics adjustment
try:
    adjusted_valuation = apply_demographics_adjustment(state_code, county_code, valuation)
    print(f"\nClimate-Aware Valuation: ${adjusted_valuation['adjusted_value']:.2f}")
    print(f"Adjustment Percentage: {adjusted_valuation['adjustment_pct']:.2%}")
    print(f"State: {adjusted_valuation['state']}")
    print(f"County: {adjusted_valuation['county']}")
    print(f"Climate Migration Trend: {adjusted_valuation['bucket']}")
    print(f"\nMigration Z-Score: {adjusted_valuation['migration z score']:.2f}")
    print(f"FEMA Risk Z-Score: {adjusted_valuation['fema risk z score']:.2f}")
    print(f"Wildfire Risk Z-Score: {adjusted_valuation['wildfire risk z score']:.2f}")
    print(f"Flood Risk Z-Score: {adjusted_valuation['flood risk z score']:.2f}")
    print(f"Heatwave Risk Z-Score: {adjusted_valuation['heatwave risk z score']:.2f}")
    
except Exception as e:
    print(f"An error occurred: {e}")
    exit(1)
#print(apply_demographics_adjustment('06', '001', 100000))
#print(provide_fips_codes('1600 Pennsylvania Avenue NW, Washington, DC'))