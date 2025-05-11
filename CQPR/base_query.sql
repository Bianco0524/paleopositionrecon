-- 相同的plate_id, age对应相同的旋转集合
WITH RECURSIVE rotation_hierarchy AS (
        (
            SELECT
                r2.plate_id,
                r2.ref_plate_id,
                r2.r1_step,
                r2.r2_step,
                r2.r1_rotation,
                r2.r2_rotation,
                points.plate_id points_plate_id,
                points.age,
                1 AS LEVEL
            FROM
                rotation_pairs_2 r2
                JOIN ( SELECT plate_id, age, COUNT ( occurrence_no ) FROM dis_points GROUP BY plate_id, age ) points ON ( r2.plate_id = points.plate_id AND ( r2.r1_step = points.age OR ( r2.r1_step < points.age AND r2.r2_step > points.age ) ) )
        )  UNION ALL (
            SELECT
                P.plate_id,
                P.ref_plate_id,
                P.r1_step,
                P.r2_step,
                P.r1_rotation,
                P.r2_rotation,
                rh.points_plate_id,
                rh.age,
                LEVEL + 1
            FROM
                rotation_hierarchy rh
                INNER JOIN rotation_pairs_2 P ON P.plate_id = rh.ref_plate_id
                AND ( P.r1_step = rh.age OR ( P.r1_step < rh.age AND P.r2_step > rh.age ) )
            )
),
RES AS ( SELECT *, RANK ( ) OVER ( PARTITION BY points_plate_id, age ORDER BY LEVEL ) AS RANK FROM rotation_hierarchy ),
Quaternion AS (
SELECT
    points_plate_id,
    age,
    rotation_multiply (
            string_agg ( plate_id :: VARCHAR, ',' ),
            string_agg ( ref_plate_id :: VARCHAR, ',' ),
            string_agg ( r1_step :: VARCHAR, ',' ),
            string_agg ( r2_step :: VARCHAR, ',' ),
            string_agg ( r1_rotation :: VARCHAR, '|' ),
            string_agg ( r2_rotation :: VARCHAR, '|' ),
            age
    ) rotation_quaternion
FROM
    RES
GROUP BY
    points_plate_id, age )
SELECT qu.points_plate_id, qu.age, dp.occurrence_no, dp.lng, dp.lat, compute_rotation(qu.rotation_quaternion, dp.lng::real, dp.lat::real) computed_coords
FROM Quaternion qu JOIN dis_points dp ON (qu.points_plate_id = dp.plate_id AND qu.age = dp.age)