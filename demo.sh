API="http://localhost:8000"
JD_FILE="sample_data/sample_jd.pdf"
RES_FILE="sample_data/sample_resume.pdf"

if [ ! -f "$JD_FILE" ] || [ ! -f "$RES_FILE" ]; then
  echo "Place a sample JD at $JD_FILE and a sample resume at $RES_FILE"
  exit 1
fi

# upload JD
echo "Uploading JD..."
JOB_ID=$(curl -s -X POST -F "file=@${JD_FILE}" -F "title=Sample Job" ${API}/upload_jd/ | jq -r .job_id)

echo "JOB_ID=$JOB_ID"

# upload resume
echo "Uploading resume..."
RES_ID=$(curl -s -X POST -F "file=@${RES_FILE}" -F "student_name=Demo Student" ${API}/upload_resume/ | jq -r .resume_id)

echo "RESUME_ID=$RES_ID"

# evaluate
echo "Evaluating..."
curl -s -X POST -F "job_id=${JOB_ID}" -F "resume_id=${RES_ID}" ${API}/evaluate/ | jq .