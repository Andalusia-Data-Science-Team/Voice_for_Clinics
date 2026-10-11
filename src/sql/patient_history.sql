SET NOCOUNT ON;
SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;

DECLARE @EndDateTime DATETIME2(0) = GETDATE();

-- Specific patient — bound as a parameter by pyodbc (the single "?" below).
-- Never string-format the patient code into this script.
DECLARE @PatientCode NVARCHAR(50) = ?;

-- Current period: last 12 months
DECLARE @CurrentStartDate DATETIME2(0) =
    DATEADD(MONTH, -12, @EndDateTime);

-- History period: another 12 months before that
DECLARE @HistoryStartDate DATETIME2(0) =
    DATEADD(MONTH, -24, @EndDateTime);


------------------------------------------------------------
-- CLEANUP
------------------------------------------------------------

DROP TABLE IF EXISTS #AllVisits;
DROP TABLE IF EXISTS #CurrentVisits;
DROP TABLE IF EXISTS #DiagnosisRows;


------------------------------------------------------------
-- 1. ALL VALID CLINICAL VISITS
--
-- 24 months of data:
--   12 months current period
--   + 12 months previous-history period
------------------------------------------------------------

SELECT
    v.ID AS VisitID,
    v.PatientID,
    v.PatientCode,
    v.ArrivalDateTime AS VisitDate,
    v.MainSpecialityEnName AS Specialty

INTO #AllVisits

FROM VisitMgt.Visit v

WHERE
    v.ArrivalDateTime IS NOT NULL

    AND v.ArrivalDateTime >= @HistoryStartDate

    AND v.ArrivalDateTime <= @EndDateTime

    AND v.PatientID IS NOT NULL

    AND v.VisitStatusID <> 3

    AND v.VisitClassificationID <> 3

    AND v.PatientCode = @PatientCode;


------------------------------------------------------------
-- INDEX
------------------------------------------------------------

CREATE CLUSTERED INDEX IX_AllVisits
ON #AllVisits
(
    PatientID,
    VisitDate,
    VisitID
);


------------------------------------------------------------
-- 2. CURRENT VISITS
--
-- Last 12 months
------------------------------------------------------------

SELECT
    VisitID,
    PatientID,
    PatientCode,
    VisitDate,
    Specialty

INTO #CurrentVisits

FROM #AllVisits

WHERE
    VisitDate >= @CurrentStartDate

    AND VisitDate <= @EndDateTime;


------------------------------------------------------------
-- INDEX
------------------------------------------------------------

CREATE CLUSTERED INDEX IX_CurrentVisits
ON #CurrentVisits
(
    PatientID,
    VisitDate,
    VisitID
);


------------------------------------------------------------
-- 3. DIAGNOSIS ROWS
--
-- ONE ROW = ONE Visit + ICD10
------------------------------------------------------------

SELECT DISTINCT

    av.VisitID,

    av.PatientID,

    av.VisitDate AS DiagnosisVisitDate,

    CAST(
        pcd.ICDDiagnoseCode AS NVARCHAR(50)
    ) AS ICDDiagnoseCode,

    CAST(
        pcd.ICDDiagnoseName AS NVARCHAR(500)
    ) AS ICDDiagnoseName

INTO #DiagnosisRows

FROM #AllVisits av

INNER JOIN PatPrlm.ProblemCard pc
    ON pc.VisitID = av.VisitID
    AND pc.PatientID = av.PatientID
    AND pc.IsDeleted = 0

INNER JOIN PatPrlm.ProblemCardDetail pcd
    ON pcd.ProblemCardID = pc.ID
    AND pcd.IsDeleted = 0

WHERE
    pcd.ICDDiagnoseCode IS NOT NULL;


------------------------------------------------------------
-- INDEX
------------------------------------------------------------

CREATE CLUSTERED INDEX IX_DiagnosisRows
ON #DiagnosisRows
(
    PatientID,
    DiagnosisVisitDate,
    VisitID
);


------------------------------------------------------------
-- 4. FINAL RESULT
--
-- ONLY REQUESTED COLUMNS
------------------------------------------------------------

SELECT

    cur.VisitID,

    cur.VisitDate,

    cur.PatientID,

    cur.PatientCode,

    cur.Specialty,

    currentDiag.CurrentICD10,

    vc.EnName AS Department,

    vt.EnName AS VisitTypeName,

    ISNULL(
        prev.PreviousVisitCount,
        0
    ) AS PreviousVisitCount


FROM #CurrentVisits cur


------------------------------------------------------------
-- DEPARTMENT
------------------------------------------------------------

INNER JOIN VisitMgt.Visit v
    ON v.ID = cur.VisitID

LEFT JOIN VisitMgt.SLKP_VisitClassification vc
    ON vc.ID = v.VisitClassificationID

LEFT JOIN VisitMgt.SLKP_VisitType vt
    ON vt.ID = v.VisitTypeID


------------------------------------------------------------
-- PREVIOUS VISITS
--
-- Count visits during the previous 12 months
-- relative to each current visit.
------------------------------------------------------------

OUTER APPLY
(
    SELECT
        COUNT(*) AS PreviousVisitCount

    FROM #AllVisits pv

    WHERE
        pv.PatientID = cur.PatientID

        AND
        (
            pv.VisitDate < cur.VisitDate

            OR

            (
                pv.VisitDate = cur.VisitDate
                AND pv.VisitID < cur.VisitID
            )
        )

        AND pv.VisitDate >=
            DATEADD(
                MONTH,
                -12,
                cur.VisitDate
            )

) prev


------------------------------------------------------------
-- CURRENT VISIT ICD-10
------------------------------------------------------------

OUTER APPLY
(
    SELECT

        STRING_AGG
        (
            CONVERT(
                NVARCHAR(MAX),
                CONCAT(
                    d.ICDDiagnoseCode,
                    ' - ',
                    d.ICDDiagnoseName
                )
            ),
            ' | '
        ) AS CurrentICD10

    FROM
    (
        SELECT DISTINCT

            dr.ICDDiagnoseCode,

            dr.ICDDiagnoseName

        FROM #DiagnosisRows dr

        WHERE
            dr.VisitID = cur.VisitID

            AND dr.ICDDiagnoseCode IS NOT NULL

    ) d

) currentDiag


------------------------------------------------------------
-- ORDER
------------------------------------------------------------

ORDER BY
    cur.VisitDate DESC,
    cur.VisitID DESC;