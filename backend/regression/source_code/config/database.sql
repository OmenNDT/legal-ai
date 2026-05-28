CREATE DATABASE logistics;
\c logistics;

-- DRIVERS
CREATE TABLE drivers (
    driver_id VARCHAR(10) PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    hire_date DATE,
    termination_date DATE,
    license_number VARCHAR(20),
    license_state CHAR(2),
    date_of_birth DATE,
    home_terminal VARCHAR(50),
    employment_status VARCHAR(20),
    cdl_class CHAR(1),
    years_experience INT
);

-- TRUCKS
CREATE TABLE trucks (
    truck_id VARCHAR(10) PRIMARY KEY,
    unit_number INT,
    make VARCHAR(30),
    model_year INT,
    vin VARCHAR(25),
    acquisition_date DATE,
    acquisition_mileage INT,
    fuel_type VARCHAR(20),
    tank_capacity_gallons INT,
    status VARCHAR(20),
    home_terminal VARCHAR(50)
);

-- TRAILERS
CREATE TABLE trailers (
    trailer_id VARCHAR(10) PRIMARY KEY,
    trailer_number INT,
    trailer_type VARCHAR(30),
    length_feet INT,
    model_year INT,
    vin VARCHAR(25),
    acquisition_date DATE,
    status VARCHAR(20),
    current_location VARCHAR(50)
);

-- CUSTOMERS
CREATE TABLE customers (
    customer_id VARCHAR(10) PRIMARY KEY,
    customer_name VARCHAR(100),
    customer_type VARCHAR(30),
    credit_terms_days INT,
    primary_freight_type VARCHAR(30),
    account_status VARCHAR(20),
    contract_start_date DATE,
    annual_revenue_potential NUMERIC(12,2)
);

-- FACILITIES
CREATE TABLE facilities (
    facility_id VARCHAR(10) PRIMARY KEY,
    facility_name VARCHAR(100),
    facility_type VARCHAR(30),
    city VARCHAR(50),
    state CHAR(2),
    latitude NUMERIC(9,4),
    longitude NUMERIC(9,4),
    dock_doors INT,
    operating_hours VARCHAR(20)
);

-- ROUTES
CREATE TABLE routes (
    route_id VARCHAR(10) PRIMARY KEY,
    origin_city VARCHAR(50),
    origin_state CHAR(2),
    destination_city VARCHAR(50),
    destination_state CHAR(2),
    typical_distance_miles INT,
    base_rate_per_mile NUMERIC(6,2),
    fuel_surcharge_rate NUMERIC(5,2),
    typical_transit_days INT
);

-- LOADS
CREATE TABLE loads (
    load_id VARCHAR(15) PRIMARY KEY,
    customer_id VARCHAR(10) REFERENCES customers(customer_id),
    route_id VARCHAR(10) REFERENCES routes(route_id),
    load_date DATE,
    load_type VARCHAR(30),
    weight_lbs INT,
    pieces INT,
    revenue NUMERIC(10,2),
    fuel_surcharge NUMERIC(10,2),
    accessorial_charges NUMERIC(10,2),
    load_status VARCHAR(20),
    booking_type VARCHAR(20)
);

-- TRIPS
CREATE TABLE trips (
    trip_id VARCHAR(15) PRIMARY KEY,
    load_id VARCHAR(15) REFERENCES loads(load_id),
    driver_id VARCHAR(10) REFERENCES drivers(driver_id),
    truck_id VARCHAR(10) REFERENCES trucks(truck_id),
    trailer_id VARCHAR(10) REFERENCES trailers(trailer_id),
    dispatch_date DATE,
    actual_distance_miles NUMERIC(10,2),
    actual_duration_hours NUMERIC(8,2),
    fuel_gallons_used NUMERIC(8,2),
    average_mpg NUMERIC(6,2),
    idle_time_hours NUMERIC(8,2),
    trip_status VARCHAR(20)
);

-- FUEL_PURCHASES
CREATE TABLE fuel_purchases (
    fuel_purchase_id VARCHAR(15) PRIMARY KEY,
    trip_id VARCHAR(15) REFERENCES trips(trip_id),
    truck_id VARCHAR(10) REFERENCES trucks(truck_id),
    driver_id VARCHAR(10) REFERENCES drivers(driver_id),
    purchase_date TIMESTAMP,
    location_city VARCHAR(50),
    location_state CHAR(2),
    gallons NUMERIC(8,2),
    price_per_gallon NUMERIC(6,3),
    total_cost NUMERIC(10,2),
    fuel_card_number VARCHAR(20)
);

-- MAINTENANCE_RECORDS
CREATE TABLE maintenance_records (
    maintenance_id VARCHAR(15) PRIMARY KEY,
    truck_id VARCHAR(10) REFERENCES trucks(truck_id),
    maintenance_date DATE,
    maintenance_type VARCHAR(30),
    odometer_reading INT,
    labor_hours NUMERIC(6,2),
    labor_cost NUMERIC(10,2),
    parts_cost NUMERIC(10,2),
    total_cost NUMERIC(10,2),
    facility_location VARCHAR(50),
    downtime_hours NUMERIC(8,2),
    service_description TEXT
);

-- DELIVERY_EVENTS
CREATE TABLE delivery_events (
    event_id VARCHAR(15) PRIMARY KEY,
    load_id VARCHAR(15) REFERENCES loads(load_id),
    trip_id VARCHAR(15) REFERENCES trips(trip_id),
    event_type VARCHAR(20),
    facility_id VARCHAR(10) REFERENCES facilities(facility_id),
    scheduled_datetime TIMESTAMP,
    actual_datetime TIMESTAMP,
    detention_minutes INT,
    on_time_flag BOOLEAN,
    location_city VARCHAR(50),
    location_state CHAR(2)
);

-- SAFETY_INCIDENTS
CREATE TABLE safety_incidents (
    incident_id VARCHAR(15) PRIMARY KEY,
    trip_id VARCHAR(15) REFERENCES trips(trip_id),
    truck_id VARCHAR(10) REFERENCES trucks(truck_id),
    driver_id VARCHAR(10) REFERENCES drivers(driver_id),
    incident_date TIMESTAMP,
    incident_type VARCHAR(50),
    location_city VARCHAR(50),
    location_state CHAR(2),
    at_fault_flag BOOLEAN,
    injury_flag BOOLEAN,
    vehicle_damage_cost NUMERIC(12,2),
    cargo_damage_cost NUMERIC(12,2),
    claim_amount NUMERIC(12,2),
    preventable_flag BOOLEAN,
    description TEXT
);

-- DRIVER_MONTHLY_METRICS
CREATE TABLE driver_monthly_metrics (
    driver_id VARCHAR(10) REFERENCES drivers(driver_id),
    month DATE,
    trips_completed INT,
    total_miles NUMERIC(10,2),
    total_revenue NUMERIC(12,2),
    average_mpg NUMERIC(6,2),
    total_fuel_gallons NUMERIC(10,2),
    on_time_delivery_rate NUMERIC(5,3),
    average_idle_hours NUMERIC(6,2),
    PRIMARY KEY (driver_id, month)
);

-- TRUCK_UTILIZATION_METRICS
CREATE TABLE truck_utilization_metrics (
    truck_id VARCHAR(10) REFERENCES trucks(truck_id),
    month DATE,
    trips_completed INT,
    total_miles NUMERIC(10,2),
    total_revenue NUMERIC(12,2),
    average_mpg NUMERIC(6,2),
    maintenance_events INT,
    maintenance_cost NUMERIC(12,2),
    downtime_hours NUMERIC(8,2),
    utilization_rate NUMERIC(5,2),
    PRIMARY KEY (truck_id, month)
);

\COPY drivers FROM '/mnt/d/Study/UIT/ML/DoAnMonHoc/source_code/data_raw/drivers.csv' CSV HEADER;
\COPY trucks FROM '/mnt/d/Study/UIT/ML/DoAnMonHoc/source_code/data_raw/trucks.csv' CSV HEADER;
\COPY trailers FROM '/mnt/d/Study/UIT/ML/DoAnMonHoc/source_code/data_raw/trailers.csv' CSV HEADER;
\COPY customers FROM '/mnt/d/Study/UIT/ML/DoAnMonHoc/source_code/data_raw/customers.csv' CSV HEADER;
\COPY facilities FROM '/mnt/d/Study/UIT/ML/DoAnMonHoc/source_code/data_raw/facilities.csv' CSV HEADER;
\COPY routes FROM '/mnt/d/Study/UIT/ML/DoAnMonHoc/source_code/data_raw/routes.csv' CSV HEADER;
\COPY loads FROM '/mnt/d/Study/UIT/ML/DoAnMonHoc/source_code/data_raw/loads.csv' CSV HEADER;
\COPY trips FROM '/mnt/d/Study/UIT/ML/DoAnMonHoc/source_code/data_raw/trips.csv' CSV HEADER;
\COPY fuel_purchases FROM '/mnt/d/Study/UIT/ML/DoAnMonHoc/source_code/data_raw/fuel_purchases.csv' CSV HEADER;
\COPY maintenance_records FROM '/mnt/d/Study/UIT/ML/DoAnMonHoc/source_code/data_raw/maintenance_records.csv' CSV HEADER;
\COPY delivery_events FROM '/mnt/d/Study/UIT/ML/DoAnMonHoc/source_code/data_raw/delivery_events.csv' CSV HEADER;
\COPY safety_incidents FROM '/mnt/d/Study/UIT/ML/DoAnMonHoc/source_code/data_raw/safety_incidents.csv' CSV HEADER;
\COPY driver_monthly_metrics FROM '/mnt/d/Study/UIT/ML/DoAnMonHoc/source_code/data_raw/driver_monthly_metrics.csv' CSV HEADER;
\COPY truck_utilization_metrics FROM '/mnt/d/Study/UIT/ML/DoAnMonHoc/source_code/data_raw/truck_utilization_metrics.csv' CSV HEADER;
