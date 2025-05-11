CREATE OR REPLACE FUNCTION quaternion_multiply(
    q1 REAL[], q2 REAL[]
) RETURNS REAL[] AS $$
DECLARE
    result REAL[];
BEGIN
    result[1] := q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3] - q1[4]*q2[4];
    result[2] := q1[1]*q2[2] + q1[2]*q2[1] + q1[3]*q2[4] - q1[4]*q2[3];
    result[3] := q1[1]*q2[3] - q1[2]*q2[4] + q1[3]*q2[1] + q1[4]*q2[2];
    result[4] := q1[1]*q2[4] + q1[2]*q2[3] - q1[3]*q2[2] + q1[4]*q2[1];

    RETURN result;
END;
$$ LANGUAGE plpgsql;

DROP FUNCTION IF EXISTS quaternion_slerp(REAL[], REAL[], DOUBLE PRECISION, DOUBLE PRECISION, DOUBLE PRECISION);
CREATE OR REPLACE FUNCTION quaternion_slerp(
    q1 REAL[], q2 REAL[], t1 DOUBLE PRECISION, t2 DOUBLE PRECISION, t_age DOUBLE PRECISION
) RETURNS REAL[] AS $$
DECLARE
    t_out DOUBLE PRECISION; -- 插值的age
    norm1 DOUBLE PRECISION; -- q1标准化参数
    norm2 DOUBLE PRECISION; -- q2标准化参数
    norm DOUBLE PRECISION; -- result标准化参数
    dot DOUBLE PRECISION; --
    result REAL[]; -- 结果四元数
    theta_0 DOUBLE PRECISION; --θ
    theta DOUBLE PRECISION; --tθ
    sin_theta DOUBLE PRECISION;
    sin_theta_0 DOUBLE PRECISION;
    s0 DOUBLE PRECISION;
    s1 DOUBLE PRECISION;
BEGIN
    t_out := (t_age-t1) / (t2-t1);
    -- q1.normalize()
    norm1 := sqrt(q1[1]*q1[1] + q1[2]*q1[2] + q1[3]*q1[3] + q1[4]*q1[4]);
    q1[1] := q1[1] / norm1;
    q1[2] := q1[2] / norm1;
    q1[3] := q1[3] / norm1;
    q1[4] := q1[4] / norm1;
    -- q2.normalize()
    norm2 := sqrt(q2[1]*q2[1] + q2[2]*q2[2] + q2[3]*q2[3] + q2[4]*q2[4]);
    q2[1] := q2[1] / norm2;
    q2[2] := q2[2] / norm2;
    q2[3] := q2[3] / norm2;
    q2[4] := q2[4] / norm2;

    dot := q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3] + q1[4]*q2[4];
    -- judge the angle greater than 9°, then inverse q2
    IF dot < 0.0 THEN
        q2[1] := -q2[1];
        q2[2] := -q2[2];
        q2[3] := -q2[3];
        q2[4] := -q2[4];
    END IF;
    -- avoid sinθ to be 0
    IF dot > 0.9995 THEN
        result[1] := q1[1] + t_out*(q2[1]-q1[1]);
        result[2] := q1[2] + t_out*(q2[2]-q1[2]);
        result[3] := q1[3] + t_out*(q2[3]-q1[3]);
        result[4] := q1[4] + t_out*(q2[4]-q1[4]);
        -- result.normalize()
        norm := sqrt(result[1]*result[1] + result[2]*result[2] + result[3]*result[3] + result[4]*result[4]);
        result[1] := result[1] / norm;
        result[2] := result[2] / norm;
        result[3] := result[3] / norm;
        result[4] := result[4] / norm;
        RETURN result;
    END IF;
    theta_0 := acos(dot);
    theta := theta_0 * t_out;
    sin_theta := sin(theta);
    sin_theta_0 := sin(theta_0);

    s0 := sin(theta_0-theta) / sin_theta_0;
    s1 := sin_theta / sin_theta_0;
    result[1] := q1[1]*s0 + q2[1]*s1;
    result[2] := q1[2]*s0 + q2[2]*s1;
    result[3] := q1[3]*s0 + q2[3]*s1;
    result[4] := q1[4]*s0 + q2[4]*s1;
    RETURN result;
END;
$$ LANGUAGE plpgsql;