# دليل الخطوات اليدوية لـ MedVoice

الدليل ده للخطوات اللي محتاجة Azure Portal أو Azure DevOps. امشي بالترتيب، وما تعملش خطوة عليها تحذير قفل قبل ما تتأكد إن الخطوة اللي قبلها نجحت.

## تشغيل المشروع بالدوكر كومبوس

المشروع بيشتغل محليًا وعلى السيرفر بنفس ملف `docker-compose.yml`. الملف فيه خدمتين: `medvoice-app` (التطبيق) و`medvoice-mongo` (قاعدة بيانات محلية للتطوير بس، مش منشورة على الهوست).

### اللي لازم يكون جاهز على السيرفر
1. **Docker Engine** و**Compose v2** متثبتين. اتأكد بالأمر: `docker compose version`.
2. **ملف البيئة** موجود على السيرفر في مسار معروف (زي `/opt/medvoice/.env`)، ومقروء لمستخدم النشر. الملف ده **مش** بيتعمل ولا بيتطبع من الـ pipeline.
3. **مجلدات قابلة للكتابة** على الهوست: `src/uploads` و`src/logs` (بيتحفظوا فيها الملفات الصوتية واللوجات، وفيها بيانات مرضى، فخليها محمية ومش متتبعة في git).
4. **الـ agent** بتاع Azure DevOps لازم يكون عليه Docker و`curl`، ويقدر يوصل للـ registry ولـ `/health/live` بتاع السيرفر.

### الأوامر بالظبط
```bash
# تشغيل (يبني الصورة لو مش موجودة)
docker compose up -d --build

# الحالة والصحة
docker compose ps

# متابعة اللوجات
docker compose logs -f medvoice-app

# فحص الصحة من الهوست
curl -s http://localhost:7777/health/live
curl -s http://localhost:7777/health

# إعادة تشغيل
docker compose restart

# إيقاف (بيسيب الـ volume بتاع الداتا)
docker compose down

# إيقاف ومع مسح الداتا (خد بالك: بيمسح قاعدة البيانات المحلية)
docker compose down -v
```

### ملاحظات مهمة
- التطبيق بيسمع على بورت **7777** جوه الكونتينر، ومربوط على `7777:7777` على الهوست.
- `medvoice-mongo` **للتطوير المحلي بس**، ومش منشور على الهوست. التطبيق بيوصله عبر شبكة الكومبوس على `mongodb://medvoice-mongo:27017`.
- الـ healthcheck بيستخدم `/health/live` (liveness بس) عشان انقطاع خدمة الـ STT ما يعملش restart للكونتينر.
- على السيرفر، الـ pipeline بيستخدم `docker compose up -d --no-build` مع صورة مسحوبة من الـ registry، مش بيبني على السيرفر.

## أ) Azure DevOps: جهّز الـ CI/CD الأول

### 1. اعمل Agent Pool (A4.1)
1. افتح Azure DevOps واختار الـ Organization ثم المشروع.
2. من **Project settings** اختار **Agent pools** ثم **Add pool**.
3. اختار نوع الـ pool المناسب للشركة، وسمّيه باسم واضح. حط الاسم في مكان `<PLACEHOLDER_SELF_HOSTED_AGENT_POOL>` في `azure-pipelines.yml`.
4. سجّل الـ self-hosted agent حسب صفحة التعليمات، وتأكد إن عليه Docker وPython و`curl` وإنه يقدر يوصل للـ registry والسيرفر الداخلي.
5. من **Pipelines > Agent pools > Security** ادّي صلاحية استخدام للـ pipeline بس.

المتوقع: الـ agent يبقى **Online** وتقدر تشغّل عليه job بسيط.
لقطة شاشة: اسم الـ pool والـ agent وحالته Online.
تحذير قفل: ما تمسحش أو توقف الـ pool القديم قبل ما تتأكد إن الـ agent الجديد Online.

### 2. اعمل Container Registry Service Connection (A4.1)
1. من **Project settings > Service connections** اختار **New service connection**.
2. اختار نوع الـ registry اللي فريقكم بيستخدمه (غالبا Azure Container Registry).
3. اختار الـ registry الصحيح، وسمّي الـ connection؛ حط الاسم في `<PLACEHOLDER_REGISTRY_SERVICE_CONNECTION>` في `azure-pipelines.yml`.
4. فعّل **Grant access permission to all pipelines** بس لو سياسة الشركة سامحة؛ الأفضل تخلي الصلاحية للـ pipeline المطلوب فقط.
5. اعمل **Verify** أو شغّل خطوة pull/push تجريبية من الـ pipeline.

المتوقع: الـ connection تبقى **Verified** والـ BuildPush يقدر يرفع image.
لقطة شاشة: اسم الاتصال وحالة Verified، من غير ما تظهر أي credentials.
تحذير قفل: ما تغيّرش connection إنتاج شغالة وما تلغيش صلاحياتها قبل ما البديل ينجح.

### 3. اعمل Variable Groups (A4.2)
1. افتح **Pipelines > Library > Variable groups > Add variable group**.
2. اعمل مجموعة لكل بيئة: dev وstaging وproduction.
3. أضف `MEDVOICE_ENV_FILE` كـ secret يشير لمسار ملف البيئة الموجود مسبقا على السيرفر، و`MEDVOICE_SMOKE_TEST_BASE_URL` كرابط داخلي للبيئة.
4. اربط كل مجموعة بالـ pipeline المناسب، وما تحطش محتوى `.env` في المتغيرات غير المشفرة أو في اللوج.
5. حط أسماء المجموعات مكان `<PLACEHOLDER_DEV_VARIABLE_GROUP>` و`<PLACEHOLDER_STAGING_VARIABLE_GROUP>` و`<PLACEHOLDER_PRODUCTION_VARIABLE_GROUP>`.

المتوقع: كل deployment job يقرأ مجموعة بيئته، وملف env موجود ومقروء على السيرفر.
لقطة شاشة: أسماء المجموعات وأسماء المتغيرات فقط، من غير القيم.
تحذير قفل: متغيّرش `MEDVOICE_ENV_FILE` في production قبل ما تتأكد إن المسار الجديد صحيح ومقروء.

### 4. اعمل Environments (A4.2، A4.3.2)
1. افتح **Pipelines > Environments > New environment**.
2. اعمل environments بالأسماء بالضبط: `dev` و`staging` و`production`.
3. اربط الـ deployment target أو الـ agent المناسب بكل بيئة.
4. في بيئة production افتح **Approvals and checks > Approvals**، أضف approvers المطلوبين، واحفظ.
5. خليك متأكد إن job الإنتاج في `azure-pipelines.yml` بيستهدف environment اسمها `production`.

المتوقع: قبل production يظهر طلب approval وما يبدأش النشر قبل الموافقة.
لقطة شاشة: صفحة production environment وصفحة approval check.
تحذير قفل: ما تحطش approval شرط على dev أو staging بالغلط؛ ده ممكن يوقف كل النشر.

### 5. Branch policy وأول Pipeline Run (A4.1، A1.5.2)
1. افتح **Repos > Branches**، اختار `main` ثم **Branch policies**.
2. أضف build validation للـ pipeline وخليه مطلوب قبل الدمج حسب سياسة الفريق.
3. افتح **Pipelines** وشغّل أول run على `main`.
4. راقب المراحل بالترتيب: CI ثم BuildPush ثم DeployDev. staging وproduction اختياريين حسب parameters والموافقات.
5. لو CI فشل، افتح **Tests** وشوف اسم الاختبار؛ ما تعملش bypass للـ policy.

المتوقع: lint وpytest ينجحوا، وبعدها الصورة تتبني وتترفع. خطوة النشر بتستخدم نفس `docker-compose.yml`: بتسحب الصورة الجديدة، بتسجّل الصورة القديمة، بتعمل `docker compose up -d --no-build` بملف البيئة الموجود على السيرفر، بتستنى الـ healthcheck، بتشغّل smoke test (`/health/live` لازم 200، و`/health` لازم 200 بحالة `ok` أو `degraded`)، ولو فشل بترجّع الصورة القديمة وتفشّل الـ job.
لقطة شاشة: ملخص run أخضر ونتائج الاختبارات.
تحذير قفل: ما تخليش build validation اختيارية على `main` لو دي بوابة الحماية المتفق عليها.

## ب) Azure Portal: الهوية والأسرار والشبكة

### 6. Key Vault والأسرار (A2.1.1، A2.1.2)
1. افتح Azure Portal وابحث عن **Key vaults > Create**.
2. اختار subscription وresource group والمنطقة المعتمدة، واكتب اسم فريد للـ vault.
3. من **Review + create** راجع الإعدادات ثم **Create**.
4. افتح الـ vault ثم **Secrets > Generate/Import**، وأضف الأسرار المطلوبة باستخدام أسماء البيئة المعرفة في `src/core/azure_identity.py`، ومنها أسماء مفاتيح Foundry وMunsit وMistral وقاعدة البيانات حسب إعدادكم.
5. فعّل `USE_KEY_VAULT` واربط `AZURE_KEY_VAULT_URL` في إعداد بيئة اختبار أولا، من غير ما تحذف أي سر من السيرفر القديم.
6. شغّل التطبيق في staging وتأكد من health وقراءة الأسرار قبل ترحيل الإنتاج.

المتوقع: حالة Key Vault سليمة والأسرار موجودة بالاسم، والتطبيق يشتغل بهوية الخدمة.
لقطة شاشة: قائمة أسماء الأسرار فقط، من غير فتح أي قيمة.
تحذير قفل: ما تمسحش أسرار `.env` أو توقف المصدر القديم قبل ما تتأكد إن Key Vault والهوية شغالين من السيرفر.

### 7. Managed Identity والأدوار (A2.2.1، A2.2.2)
1. افتح مورد التطبيق/السيرفر ثم **Identity**، وفعل **System assigned** أو اختار الهوية المعتمدة.
2. احفظ، وسجّل **Object (principal) ID**.
3. في Key Vault افتح **Access control (IAM) > Add role assignment**، اختار **Key Vault Secrets User**، ثم اختار الهوية.
4. كرر على Foundry resource واختار **Cognitive Services User** أو الدور المتفق عليه باسم Foundry في المؤسسة.
5. اختبر التطبيق من السيرفر باستخدام Entra/Managed Identity قبل أي تغيير في المصادقة.
6. ملاحظة: الكود الحالي يستخدم `DefaultAzureCredential`، لكنه لا يقرأ شهادة مخصصة من `AZURE_CLIENT_CERTIFICATE_PATH`. لو المؤسسة اشترطت شهادة App Registration، لازم يتنفذ دعم الشهادة ويُختبر قبل حذف client secret.

المتوقع: الدورين ظاهرين للهوية الصحيحة، وطلب token من السيرفر ناجح.
لقطة شاشة: role assignments مع اسم الهوية والأدوار.
تحذير قفل: ما تشيلش API-key auth من Foundry قبل ما تثبت إن Entra token شغال فعليا من نفس السيرفر ونفس الشبكة.

### 8. Foundry Networking وAPI-key auth (A2.5.1، A2.5.2)
1. في Foundry resource افتح **Networking**، وراجع وضع Public network access.
2. أضف عناوين الخروج الداخلية وعناوين AWS المعتمدة حسب ما يديها IT، واحفظ.
3. كرر الإعداد على Key Vault إذا كان مطلوبا.
4. اختبر من السيرفر المسموح ومن اتصال غير مسموح، وتأكد إن غير المسموح بياخد 403.
5. بعد نجاح Entra من السيرفر فقط: افتح Foundry resource > **Settings/Authentication**، عطّل local/API-key authentication، واحفظ.
6. شغّل `scripts/check_entra_only_auth.py` من بيئة مسموح لها؛ المتوقع API key مرفوض وEntra token ناجح.

المتوقع: الاتصال المسموح يعمل، والاتصال من IP غير مسموح مرفوض؛ Entra يعمل وAPI key مرفوض بعد التعطيل.
لقطة شاشة: إعداد الشبكة ونتيجة اختبار الرفض والقبول من غير عرض أي token/key.
تحذير قفل شديد: الشبكة الغلط أو تعطيل API key قبل إثبات Entra ممكن يمنع كل الفريق من الوصول. خلي جلسة مسؤول ثانية وطريقة استرجاع جاهزة، وما تنفذش الخطوة الأخيرة لو اختبار token مش PASS.

## ج) الحصص وتشغيل الاختبار

### 9. TPM/RPM للتوزيعات (A3.2)
1. افتح Azure AI Foundry ثم المورد ثم **Deployments**.
2. افتح كل deployment من الستة: `gpt-5.6-luna`، `gpt-5.6-terra`، `gpt-5.6-sol`، `gpt-4.1-mini`، `DeepSeek-V3.2`، `model-router`.
3. سجّل قيم TPM وRPM الظاهرة في صفحة كل deployment.
4. اكتب القيم في جدول `docs/foundry_models.md`. املأ متغير `FOUNDRY_QUOTA_TPM__<DEPLOYMENT>` المناسب من نفس الصفحة. ما تخمّنش أي رقم.

المتوقع: الست صفوف فيها الأرقام الفعلية من البوابة.
لقطة شاشة: صفحة quotas لكل deployment أو جدول موحد.
تحذير: ما تزودش quota على حساب deployment تاني من غير موافقة صاحب الخدمة.

### 10. شغّل smoke test (A3.3)
1. من نسخة المشروع المعتمدة، فعّل البيئة الافتراضية.
2. تأكد إن إعدادات Foundry مضبوطة في البيئة من غير ما تطبعها.
3. شغّل `python scripts/smoke_test_deployments.py` مرة واحدة.
4. احتفظ بملخص الأعداد وPASS/FAIL فقط؛ ما تنسخش prompts أو الردود.
5. حدّث `docs/foundry_models.md` بنتيجة JSON وcontent-filter لكل deployment.

المتوقع: تقرير counts فقط؛ راجع أي FAIL مع مسؤول Foundry.
لقطة شاشة: جدول النتائج أو terminal summary، من غير أسرار أو نصوص سريرية.

## د) الموافقات المطلوبة

### 11. التوقيعات
- **Clinical Lead:** افتح `src/prompts/routing_criteria.txt`، راجع المعايير العامة، ثم سجّل الموافقة والتاريخ في مسار الموافقات المعتمد. يقفل شرط اعتماد معيار التوجيه.
- **Architect:** راجع `LLM_TOTAL_TIMEOUT_S=300` وهل هو حد retry deadline المناسب. لا تغيّر الرقم من غير قرار موثق.
- **Compliance:** راجع data residency للتوزيعات الستة. حسب قرار المالك، كل التوزيعات Global Standard؛ أكد إن ده مناسب للسياسة قبل الإنتاج.

المتوقع: سجل موافقة رسمي لكل جهة.
لقطة شاشة: صفحة الموافقة أو رقم الطلب.

## هـ) أسئلة لـ IT قبل الإنتاج

1. هل السيرفر يقدر يوصل PyPI؟ لو لأ، إيه عنوان الـ internal package mirror وطريقة استخدامه؟
2. إيه egress IPs الفعلية للسيرفر/الـ container اللي لازم تتضاف لـ Foundry وKey Vault؟
3. إيه IPs الخاصة بـ Azure DevOps agents؟ وهل بتتغير؟
4. هل الـ agent يقدر يوصل للـ registry والسيرفر الداخلي على المنفذ المتفق عليه؟
5. مين مسؤول عن استرجاع الوصول لو firewall أو Entra policy منعت تسجيل الدخول؟

اكتب الإجابات في ticket داخلي، من غير مفاتيح أو tokens.