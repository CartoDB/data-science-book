def get_nyc_bridge_connections():
    return {
        # Verrazzano-Narrows Bridge (Staten Island <-> Brooklyn)
        4551: [2571],
        2571: [4551],
        # Williamsburg Bridge (Manhattan <-> Brooklyn)
        2923: [3498],
        3498: [2923],
        # Queensborough Bridge (Manhattan <-> Queens)
        3595: [3900],
        3900: [3736, 3595],
        # Roosevelt Island <-> Queens
        3736: [3900],
        # Astoria <-> East Harlem
        3943: [3688],
        3688: [2028, 3943],
        # East Harlem <-> Bronx
        2028: [3688],
        # Northern Manhattan <-> Marble Hill
        3769: [2237],
        2237: [3769],
        # Rockaways
        4411: [3009],
        3009: [4411],
    }
