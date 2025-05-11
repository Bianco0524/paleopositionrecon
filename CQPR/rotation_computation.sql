DROP FUNCTION IF EXISTS compute_rotation(text[], DOUBLE PRECISION, DOUBLE PRECISION);
CREATE OR REPLACE FUNCTION compute_rotation(
    base_text text[], lng DOUBLE PRECISION, lat DOUBLE PRECISION
) RETURNS REAL[] AS $$
DECLARE
    base REAL[];
    v0 REAL[];
    v1 REAL[];
    con REAL[];
    res REAL[];
    r_lng DOUBLE PRECISION := radians(lng); -- radian of lng
    r_lat DOUBLE PRECISION := radians(lat); -- radian of lat
BEGIN
    -- transformation
    base[1] := base_text[1]::REAL;
    base[2] := base_text[2]::REAL;
    base[3] := base_text[3]::REAL;
    base[4] := base_text[4]::REAL;
    -- vector
    v0[1] := 0;
    v0[2] := cos(r_lat) * cos(r_lng);
    v0[3] := cos(r_lat) * sin(r_lng);
    v0[4] := sin(r_lat);
    -- conjugate
    con[1] := base[1];
    con[2] := -base[2];
    con[3] := -base[3];
    con[4] := -base[4];
    --rotate by quaternion
    v1 := quaternion_multiply(quaternion_multiply(base, v0), con);
    res[1] := v1[2];
    res[2] := v1[3];
    res[3] := v1[4];
    -- vector to lng & lat
    RETURN array[degrees(atan2(res[2], res[1])), degrees(asin(res[3]))];
END;
$$ LANGUAGE plpgsql;

DROP FUNCTION IF EXISTS rotation_multiply(TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, DOUBLE PRECISION);
CREATE OR REPLACE FUNCTION rotation_multiply(
    plate_ids TEXT, ref_plate_ids TEXT, r1_steps TEXT, r2_steps TEXT, r1_rotations TEXT, r2_rotations TEXT, age DOUBLE PRECISION
) RETURNS TEXT[] AS $$
DECLARE
    plate_ids_array TEXT[];
    ref_plate_ids_array TEXT[];
    r1_step_array TEXT[];
    r2_step_array TEXT[];
    r1_rotation_array TEXT[];
    r2_rotation_array TEXT[];
    length_array INT;
    temp_loc INT;
    base REAL[] := array[1,0,0,0];
    res REAL[];
BEGIN
    -- get array from text
    plate_ids_array := reverse_array(string_to_array(plate_ids, ','));
    ref_plate_ids_array := reverse_array(string_to_array(ref_plate_ids, ','));
    r1_step_array := reverse_array(string_to_array(r1_steps, ','));
    r2_step_array := reverse_array(string_to_array(r2_steps, ','));
    r1_rotation_array := reverse_array(string_to_array(r1_rotations, '|'));
    r2_rotation_array := reverse_array(string_to_array(r2_rotations, '|'));
    length_array := array_length(r2_rotation_array, 1);
    -- multiply
    FOR temp_loc IN 1..length_array LOOP
        IF r1_step_array[temp_loc]::DOUBLE PRECISION = age THEN
            res := euler_to_quaternion(r1_rotation_array[temp_loc]);
            base := quaternion_multiply(base, res);
        ELSE
            res := quaternion_slerp(euler_to_quaternion(r1_rotation_array[temp_loc]),
                                    euler_to_quaternion(r2_rotation_array[temp_loc]),
                                    r1_step_array[temp_loc]::DOUBLE PRECISION, r2_step_array[temp_loc]::DOUBLE PRECISION, age);
            base := quaternion_multiply(base, res);
        END IF;
    END LOOP;
    RETURN base;
END;
$$ LANGUAGE plpgsql;