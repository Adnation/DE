SELECT AVG(trips.actual_eta) - 
(SELECT
(
 (SELECT MAX(actual_eta) FROM
   (
   SELECT TOP 50 PERCENT actual_eta 
   FROM trips INNER JOIN cities ON trips.city_id=cities.city_id
   WHERE 
	cities.city_name IN ('Pembroke', 'Stamford') AND
	trips.departure_date BETWEEN (CURDATE() - INTERVAL 360 DAY) AND CURDATE() AND
	trips.status = 'completed'
   ORDER BY actual_eta))
 +
 (SELECT MIN(actual_eta) FROM
   (
   	SELECT TOP 50 PERCENT actual_eta
    FROM trips INNER JOIN cities ON trips.city_id=cities.city_id
    WHERE 
	cities.city_name IN ('Pembroke', 'Stamford') AND
	trips.departure_date BETWEEN (CURDATE() - INTERVAL 360 DAY) AND CURDATE() AND
	trips.status = 'completed'
    ORDER BY actual_eta DESC))
) / 2), 

AVG(trips.predicted_eta) - 
(SELECT
(
 (SELECT MAX(predicted_eta) FROM
   (
   SELECT TOP 50 PERCENT predicted_eta 
   FROM trips INNER JOIN cities ON trips.city_id=cities.city_id
   WHERE 
	cities.city_name IN ('Pembroke', 'Stamford') AND
	trips.departure_date BETWEEN (CURDATE() - INTERVAL 360 DAY) AND CURDATE() AND
	trips.status = 'completed'
   ORDER BY predicted_eta))
 +
 (SELECT MIN(predicted_eta) FROM
   (
   	SELECT TOP 50 PERCENT predicted_eta
    FROM trips INNER JOIN cities ON trips.city_id=cities.city_id
    WHERE 
	cities.city_name IN ('Pembroke', 'Stamford') AND
	trips.departure_date BETWEEN (CURDATE() - INTERVAL 360 DAY) AND CURDATE() AND
	trips.status = 'completed'
    ORDER BY predicted_eta DESC))
) / 2)
FROM trips