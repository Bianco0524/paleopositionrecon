DROP FUNCTION IF EXISTS euler_to_quaternion(TEXT);
CREATE OR REPLACE FUNCTION euler_to_quaternion(
    euler_text TEXT
) RETURNS REAL[] AS $$
DECLARE
    euler REAL[];
    lng DOUBLE PRECISION;
    lat DOUBLE PRECISION;
    angle DOUBLE PRECISION;
    q REAL[];
BEGIN
    -- split to array from text 去掉首尾{}，按逗号划分
    euler := string_to_array(regexp_replace(euler_text, '[{}]', '', 'g'), ',');
    -- get lat, lng, angle from array
    lat := radians(euler[1]);
    lng := radians(euler[2]);
    angle := radians(euler[3]);
    -- translation
    q[1] := cos(angle/2);
    q[2] := cos(lat) * cos(lng) * sin(angle/2);
    q[3] := cos(lat) * sin(lng) * sin(angle/2);
    q[4] := sin(lat) * sin(angle/2);
    RETURN q;
END;
$$ LANGUAGE plpgsql;

-- 数组倒序，避免反向 --
CREATE OR REPLACE FUNCTION reverse_array(
        aa TEXT[]
) RETURNS TEXT[] AS $$
DECLARE
        reversed_array TEXT[];
        i INT;
        length_array INT;
BEGIN
        IF aa IS NULL OR array_length(aa, 1) IS NULL THEN
     RETURN '{}';
  END IF;
        length_array := array_length(aa, 1);
        reversed_array := ARRAY[]::TEXT[];

        FOR i IN 1..length_array LOOP
    reversed_array := reversed_array || ARRAY[aa[length_array - i + 1]];
  END LOOP;

        RETURN reversed_array;
END;
$$ LANGUAGE plpgsql;