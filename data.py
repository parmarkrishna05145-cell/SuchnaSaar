"""Corpus and reference facts for SuchnaSaar.

Two things live here.

1.  SENTENCES -- a hand-authored corpus of sentences written in the register
    of Indian government notices (gazette notifications, departmental
    circulars, scheme guidelines, public notices). Every sentence carries one
    of three roles:

        reference         background, preamble, legal boilerplate, or a
                          statement of something that has already happened.
                          Reading it changes nothing about what you must do.
        action_required   tells a named actor to do something: submit a
                          document, apply, link a record, pay a fee.
        deadline          fixes a date or a period within which something
                          must be done.

    These sentences were written for this project. They imitate the *style*
    of Indian official drafting; they are not quoted from any gazette. That
    is stated plainly in the report, because a corpus of imitated drafting
    has to be honest about what it is.

2.  NOTICES -- five complete synthetic notices used by the demo screen and
    by the screenshot deliverable. Each one is fictional but assembled from
    the same sentence patterns, so the classifier sees the register it was
    trained on.

3.  SCHEMES and HELPLINES -- real, verified facts only. Every entry carries
    the official domain it was checked against. Nothing here is invented;
    if a fact could not be verified it was left out rather than guessed.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Sentence-level corpus
# --------------------------------------------------------------------------
LABELS = ["reference", "action_required", "deadline"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}

SENTENCES: list[tuple[str, str]] = [
    # ---------------------------------------------------------------- reference
    ("The Scheme was launched to provide income support to landholding farmer families.", "reference"),
    ("This notification is issued in supersession of the circular issued earlier on the same subject.", "reference"),
    ("The Ministry of Finance has approved the revised guidelines of the Scheme.", "reference"),
    ("The objective of the Scheme is to extend timely and transparent assistance through direct benefit transfer.", "reference"),
    ("In exercise of the powers conferred by sub-section (1) of section 3 of the Act, the Central Government hereby makes the following rules.", "reference"),
    ("The Authority has already issued detailed instructions to all field offices in this regard.", "reference"),
    ("Keeping in view the representations received from stakeholders, the following clarifications are issued.", "reference"),
    ("The Scheme has been extended for a further period of five years with the same terms and conditions.", "reference"),
    ("A total of 2.4 crore beneficiaries have been covered under the Scheme so far.", "reference"),
    ("The portal has been developed by the National Informatics Centre.", "reference"),
    ("The provisions of this order shall come into force from the date of its publication in the Official Gazette.", "reference"),
    ("It has been decided to constitute a committee to examine the matter in detail.", "reference"),
    ("The Committee shall consist of a Chairperson and four members.", "reference"),
    ("The quorum for the meeting of the Committee shall be three members.", "reference"),
    ("The minutes of the proceedings shall be recorded and placed before the Authority.", "reference"),
    ("This circular is applicable to all scheduled commercial banks and co-operative banks.", "reference"),
    ("The guidelines shall mutatis mutandis apply to all subordinate offices.", "reference"),
    ("The definition of family for the purposes of the Scheme is as given in the Annexure.", "reference"),
    ("The details of the Scheme are available on the official website of the Department.", "reference"),
    ("All stakeholders are requested to note the above for compliance.", "reference"),
    ("This is for information of all concerned.", "reference"),
    ("For record only. No acknowledgement is necessary.", "reference"),
    ("No action is required on the part of the addressee.", "reference"),
    ("No further action is required from the applicant at this stage.", "reference"),
    ("Nothing further is required to be done by the beneficiary.", "reference"),
    ("The matter is under examination and a decision will be communicated in due course.", "reference"),
    ("The earlier order stands modified to the extent indicated in the Annexure.", "reference"),
    ("The corrigendum is issued to correct a clerical error in the original notification.", "reference"),
    ("The figures mentioned in paragraph 4 may be read as figures for the preceding financial year.", "reference"),
    ("The scheme was introduced with a view to strengthening the digital public infrastructure of the country.", "reference"),
    ("The Department has sanctioned an outlay for the implementation of the Scheme.", "reference"),
    ("The State Government shall be responsible for the implementation of the Scheme at the field level.", "reference"),
    ("The nodal officer for the Scheme shall be the Director of the concerned Department.", "reference"),
    ("The application forms are available free of cost at the office of the Block Development Officer.", "reference"),
    ("The scheme guidelines provide for an annual review by the Department.", "reference"),
    ("The Authority has noted with satisfaction the progress made in the implementation of the Scheme.", "reference"),
    ("This order shall not apply to establishments in the unorganised sector.", "reference"),
    ("Persons covered under the earlier scheme shall continue to be governed by its provisions.", "reference"),
    ("The decision of the Authority shall be final and binding on all parties concerned.", "reference"),
    ("The hearing in the matter has been concluded and the order is reserved.", "reference"),
    ("The land records of the concerned district have been digitised and integrated with the portal.", "reference"),
    ("The number of instalments released under the Scheme has increased over the years.", "reference"),
    ("The list of beneficiaries was prepared on the basis of the data received from the States.", "reference"),
    ("The Department has clarified that the earlier instructions continue to hold good.", "reference"),
    ("The provisions of the Act are in addition to, and not in derogation of, any other law in force.", "reference"),
    ("The term of office of the members shall be three years from the date of appointment.", "reference"),
    ("The meeting was convened under the chairpersonship of the Secretary.", "reference"),
    ("The above position was reiterated in the review meeting held with the States.", "reference"),
    ("The matter was considered by the competent authority and the following decision has been taken.", "reference"),
    ("It is clarified that the Scheme is implemented through the existing administrative machinery.", "reference"),
    ("The expenditure on the Scheme shall be met from the budget provision of the Department.", "reference"),
    ("The scheme has been designed to be transparent, paperless and faceless.", "reference"),
    ("This office has no objection to the proposal as submitted.", "reference"),
    ("The application was forwarded to the sanctioning authority for consideration.", "reference"),
    ("The records of the case are available for perusal in the office of the undersigned.", "reference"),
    ("The document is issued without any prejudice to the rights of the parties.", "reference"),
    ("The Department has received a reference from the State Government on this matter.", "reference"),
    ("The guidelines are available in English and in Hindi on the departmental portal.", "reference"),
    ("The present notification supersedes all previous notifications on the subject.", "reference"),
    ("The scheme is funded entirely by the Central Government.", "reference"),
    ("A grievance redressal mechanism has been put in place for the beneficiaries.", "reference"),
    ("The Committee has recommended that the existing arrangement may continue.", "reference"),
    ("The Department has decided to extend the facility to all districts in a phased manner.", "reference"),
    ("The scheme shall be implemented in mission mode through the existing institutions.", "reference"),
    ("The progress of implementation will be reviewed periodically by the Department.", "reference"),
    ("The scheme is being implemented with the active participation of local bodies.", "reference"),
    ("The names of the empanelled agencies are displayed on the portal.", "reference"),
    ("The Authority has taken on record the compliance report submitted by the entity.", "reference"),
    ("The matter is sub judice and no comments can be offered at this stage.", "reference"),
    ("The above guidelines were circulated to all concerned for information.", "reference"),
    ("The Department has no objection to the deviation proposed in the design.", "reference"),
    ("The scheme was earlier known by a different name.", "reference"),
    ("The benefit under the Scheme is in addition to any other benefit admissible under any other scheme.", "reference"),

    # ---------------------------------------------------------- action_required
    ("Applicants are required to submit a self-attested copy of the Aadhaar card along with the application.", "action_required"),
    ("The beneficiary must upload a copy of the land record on the portal.", "action_required"),
    ("All registered entities shall furnish a declaration of compliance to the Authority.", "action_required"),
    ("Applicants are directed to link their bank account with the Aadhaar number.", "action_required"),
    ("The applicant is required to complete the e-KYC process before the benefit can be released.", "action_required"),
    ("Candidates shall produce the original documents at the time of verification.", "action_required"),
    ("The establishment must file a fresh registration application with the Department.", "action_required"),
    ("Farmers are advised to update their mobile number in the records of the Department.", "action_required"),
    ("The applicant shall enclose a certificate from the competent authority.", "action_required"),
    ("Entities are required to submit the annual return in the prescribed format.", "action_required"),
    ("The member shall apply for the withdrawal of the accumulated balance in Form 19.", "action_required"),
    ("Traders must obtain a licence from the Municipal Corporation before commencing business.", "action_required"),
    ("The applicant is required to pay the prescribed fee through the online challan.", "action_required"),
    ("Beneficiaries shall submit a self-declaration stating that they do not hold any other land.", "action_required"),
    ("The Society must register itself with the Registrar of Societies.", "action_required"),
    ("The applicant shall affix a recent passport size photograph on the application form.", "action_required"),
    ("You are required to update your address in the Aadhaar records.", "action_required"),
    ("The employer shall deposit the contribution with the fund within the prescribed period.", "action_required"),
    ("Applicants must verify the details entered in the application before final submission.", "action_required"),
    ("The dealer shall display the registration certificate at a prominent place in the premises.", "action_required"),
    ("Citizens are advised to keep the acknowledgement number safe for future reference.", "action_required"),
    ("The applicant shall submit an affidavit duly attested by a notary.", "action_required"),
    ("All landholding families must complete the seeding of their Aadhaar with the bank account.", "action_required"),
    ("The candidate must appear for the physical verification of documents at the designated centre.", "action_required"),
    ("The entity shall furnish the details of its directors to the Authority.", "action_required"),
    ("Applicants are required to attach the income certificate issued by the Tehsildar.", "action_required"),
    ("The household must register at the nearest Common Service Centre.", "action_required"),
    ("The applicant shall obtain a no objection certificate from the landlord.", "action_required"),
    ("You must correct the mismatch in your name in the records of the Department.", "action_required"),
    ("The organisation shall submit a utilisation certificate for the grant received.", "action_required"),
    ("The applicant is required to submit a copy of the bank passbook.", "action_required"),
    ("Beneficiaries are directed to complete their biometric authentication at the ration shop.", "action_required"),
    ("The taxpayer shall file the return of income for the assessment year.", "action_required"),
    ("Applicants must provide a valid mobile number and email address for communication.", "action_required"),
    ("The owner shall produce the original registration certificate for inspection.", "action_required"),
    ("Students are required to upload the marksheet of the qualifying examination.", "action_required"),
    ("The applicant shall fill in all the mandatory fields marked with an asterisk.", "action_required"),
    ("Establishments must obtain prior approval of the competent authority for any change of address.", "action_required"),
    ("The beneficiary shall open an account in a bank branch near the place of residence.", "action_required"),
    ("Applicants are required to sign the declaration at the bottom of the form.", "action_required"),
    ("The unit shall install the prescribed measuring device at its own cost.", "action_required"),
    ("The applicant must submit the application through the online portal only.", "action_required"),
    ("The transporter shall produce a valid permit for the vehicle.", "action_required"),
    ("Persons covered under the Scheme shall update their bank details on the portal.", "action_required"),
    ("The applicant shall ensure that the photograph uploaded is not blurred.", "action_required"),
    ("The contractor must furnish a performance guarantee in the prescribed form.", "action_required"),
    ("Citizens are advised to change the password of their account if it has been compromised.", "action_required"),
    ("The applicant shall submit the caste certificate issued by the competent authority.", "action_required"),
    ("The dealer must maintain a register of all sales for inspection.", "action_required"),
    ("Applicants are required to respond to the query raised by the processing officer.", "action_required"),
    ("The school shall upload the attendance records of the students on the portal.", "action_required"),
    ("The borrower shall submit proof of utilisation of the loan amount.", "action_required"),
    ("The applicant must obtain a digital signature certificate for filing online.", "action_required"),
    ("All concerned shall comply with the directions contained in this order.", "action_required"),
    ("The beneficiary shall give a declaration that the land is under cultivation.", "action_required"),
    ("The applicant is required to appear in person before the registering authority.", "action_required"),
    ("The firm shall notify any change in its constitution to the Department.", "action_required"),
    ("Applicants should ensure that the scanned documents are legible.", "action_required"),
    ("The owner shall surrender the old licence before a duplicate is issued.", "action_required"),
    ("The institution must submit a statement of accounts audited by a chartered accountant.", "action_required"),
    ("You are requested to submit the pending documents at the earliest.", "action_required"),
    ("The applicant shall declare that no similar application is pending before any other authority.", "action_required"),
    ("The member shall nominate a person to receive the benefits under the Scheme.", "action_required"),
    ("Vendors must register on the Government e-Marketplace to participate in the tender.", "action_required"),
    ("The applicant shall submit the proof of residence such as a utility bill or rent agreement.", "action_required"),
    ("The establishment shall display the abstract of the Act at the main entrance.", "action_required"),
    ("The applicant is required to make the payment through the designated payment gateway.", "action_required"),
    ("Concerned officers shall verify the applications within the stipulated period.", "action_required"),
    ("The beneficiary must activate the account before the benefit is credited.", "action_required"),
    ("Applicants shall keep a copy of the submitted application for their records.", "action_required"),

    # --------------------------------------------------------------- deadline
    ("The application must be submitted within 30 days from the date of publication of this notice.", "deadline"),
    ("Objections, if any, may be filed within 15 days from the date of this notification.", "deadline"),
    ("The last date for submission of the application is 31 December.", "deadline"),
    ("Applications received after the cut-off date will not be entertained.", "deadline"),
    ("The entity shall furnish the information not later than 7 days from the date of receipt of this letter.", "deadline"),
    ("The return shall be filed on or before the due date.", "deadline"),
    ("The prescribed time limit for filing the appeal is 45 days.", "deadline"),
    ("Claims must be submitted latest by 15 March.", "deadline"),
    ("The response should reach this office within 10 days of the issue of this notice.", "deadline"),
    ("The document shall be produced within 21 days of the receipt of this communication.", "deadline"),
    ("The registration shall be valid up to 31 March of the following year.", "deadline"),
    ("The last date for the payment of the fee is 28 February.", "deadline"),
    ("The verification shall be completed within 45 days.", "deadline"),
    ("The applicant shall appear before the authority within 7 days.", "deadline"),
    ("The hearing is scheduled on 12 January at 11:00 hours.", "deadline"),
    ("The deadline for the submission of the utilisation certificate is 30 June.", "deadline"),
    ("The extension of time is granted up to 30 September.", "deadline"),
    ("The time limit for compliance is extended till 31 July.", "deadline"),
    ("The objections must be received within 14 days of the publication of this notice.", "deadline"),
    ("The applicant shall rectify the defects within 7 days of the communication of the deficiency.", "deadline"),
    ("The window for the correction of the application will remain open for 5 days.", "deadline"),
    ("The scheme shall remain in force up to 31 March.", "deadline"),
    ("The contribution for the month shall be deposited within 15 days of the close of the month.", "deadline"),
    ("The last date for the linking of the Aadhaar number with the bank account is 30 April.", "deadline"),
    ("The licence shall expire on 31 December unless renewed.", "deadline"),
    ("The period of limitation for filing the review petition is 30 days.", "deadline"),
    ("The appeal shall be preferred within 60 days from the date of the order.", "deadline"),
    ("The notice shall be complied with within 3 weeks.", "deadline"),
    ("The applicant must submit the pending document within 2 days of this intimation.", "deadline"),
    ("The online window closes on 20 February at 23:59 hours.", "deadline"),
    ("The verification camp shall be held from 5 January to 20 January.", "deadline"),
    ("The last date for the deposit of the amount is 31 October.", "deadline"),
    ("The renewal application must be made at least 30 days before the date of expiry.", "deadline"),
    ("The information shall be furnished within 24 hours of the occurrence of the event.", "deadline"),
    ("The applicant shall be given 15 days to file the reply.", "deadline"),
    ("The final date for the registration is 15 November.", "deadline"),
    ("The time limit prescribed under the rules is 90 days.", "deadline"),
    ("The claims received after 28 February shall stand rejected.", "deadline"),
    ("The period of the scheme has been extended by 6 months.", "deadline"),
    ("The compliance report is to be submitted within 30 days of the close of the financial year.", "deadline"),
    ("The last date of submission has been extended up to 31 August.", "deadline"),
    ("The demand shall be paid within 30 days of the service of this notice.", "deadline"),
    ("The applicant shall complete the process before the due date to avoid the late fee.", "deadline"),
    ("The objections will be considered only if received before the date fixed for the hearing.", "deadline"),
    ("The tenure of the licence is 5 years from the date of issue.", "deadline"),
    ("The prescribed period for the completion of the enquiry is 6 months.", "deadline"),
    ("The application shall be submitted not earlier than 60 days before the commencement of the course.", "deadline"),
    ("The time allowed for filing the written statement is 4 weeks.", "deadline"),
    ("The due date for the payment of the instalment is 10 January.", "deadline"),
    ("The candidates must report by 09:30 hours on the day of the examination.", "deadline"),
    ("The documents shall be verified within 3 working days of receipt.", "deadline"),
    ("The last date for applying for the correction window is 5 March.", "deadline"),
    ("The tenant shall vacate the premises on or before 31 March.", "deadline"),
    ("The report shall be submitted within a period of 2 months from the date of this order.", "deadline"),
    ("The amnesty period shall remain open till 31 December.", "deadline"),
    ("The objections shall be filed within 30 days of the date of the public notice.", "deadline"),
    ("The amount must be remitted within 7 working days.", "deadline"),
    ("The final opportunity is granted up to 15 February.", "deadline"),
    ("The enrolment shall be completed within 3 months of the date of joining.", "deadline"),
    ("The bid shall be submitted before 15:00 hours on 20 March.", "deadline"),
    ("The warranty period shall be 12 months from the date of installation.", "deadline"),
    ("The application window shall be open for a period of 21 days.", "deadline"),
    ("The last date for the submission of the hard copy is 10 April.", "deadline"),
    ("The claimant shall produce the documents within 45 days of the notice.", "deadline"),
]

# --------------------------------------------------------------------------
# Complete synthetic notices used by the demo screen
# --------------------------------------------------------------------------
NOTICES = [
    {
        "id": "N1",
        "title": "Public Notice: Updation of Aadhaar-linked bank details for benefit transfer",
        "issuer": "Department of Agriculture and Farmers' Welfare",
        "subject": "Seeding of Aadhaar with bank account for release of instalment",
        "body": (
            "The Scheme was launched to provide income support to landholding farmer families. "
            "The Department has already released several instalments through direct benefit transfer. "
            "The Ministry of Finance has approved the revised guidelines of the Scheme. "
            "It has been observed that in a number of cases the benefit could not be credited because the bank account is not linked with the Aadhaar number. "
            "Beneficiaries are directed to link their bank account with the Aadhaar number. "
            "The applicant is required to complete the e-KYC process before the benefit can be released. "
            "Applicants are required to submit a self-attested copy of the Aadhaar card along with the application. "
            "The last date for the linking of the Aadhaar number with the bank account is 30 April. "
            "The application must be submitted within 30 days from the date of publication of this notice. "
            "The documents shall be verified within 3 working days of receipt. "
            "The beneficiary must upload a copy of the land record on the portal. "
            "The details of the Scheme are available on the official website of the Department. "
            "The Scheme has been extended for a further period of five years with the same terms and conditions. "
            "This is for information of all concerned."
        ),
    },
    {
        "id": "N2",
        "title": "Office Memorandum: Filing of annual return and payment of prescribed fee",
        "issuer": "Regional Office, Employees' State Insurance Corporation",
        "subject": "Annual return for the preceding financial year and payment of contribution",
        "body": (
            "In exercise of the powers conferred by the provisions of the Act, the Regional Office hereby directs as follows. "
            "The establishment must file a fresh registration application with the Department where the earlier registration has lapsed. "
            "The employer shall deposit the contribution with the fund within the prescribed period. "
            "The contribution for the month shall be deposited within 15 days of the close of the month. "
            "Entities are required to submit the annual return in the prescribed format. "
            "The establishment shall display the abstract of the Act at the main entrance. "
            "The institution must submit a statement of accounts audited by a chartered accountant. "
            "The compliance report is to be submitted within 30 days of the close of the financial year. "
            "The last date for submission of the annual return is 31 December. "
            "Applications received after the cut-off date will not be entertained. "
            "The prescribed time limit for filing the appeal is 45 days. "
            "The applicant is required to pay the prescribed fee through the online challan. "
            "The provisions of the Act are in addition to, and not in derogation of, any other law in force. "
            "No action is required on the part of the addressee in respect of the earlier communication."
        ),
    },
    {
        "id": "N3",
        "title": "Public Notice: Invitation of objections to draft electoral roll revision",
        "issuer": "Office of the Electoral Registration Officer, District Administration",
        "subject": "Draft publication of electoral roll and invitation of claims and objections",
        "body": (
            "Notice is hereby given that the draft electoral roll has been published in accordance with the directions of the Commission. "
            "The draft roll has been prepared on the basis of the qualifying date. "
            "All stakeholders are requested to note the above for compliance. "
            "Objections, if any, may be filed within 15 days from the date of this notification. "
            "The objections must be received within 14 days of the publication of this notice. "
            "Applicants are required to submit the proof of residence such as a utility bill or rent agreement. "
            "The applicant shall enclose a certificate from the competent authority where the address has changed. "
            "Forms are available free of cost at the office of the Electoral Registration Officer. "
            "The records of the case are available for perusal in the office of the undersigned. "
            "The hearing in the matter has been concluded and the order is reserved. "
            "This is for information of all concerned."
        ),
    },
    {
        "id": "N4",
        "title": "Circular: Corrigendum to the grant utilisation guidelines",
        "issuer": "Department of Rural Development",
        "subject": "Correction of a clerical error in the earlier circular and clarification of timelines",
        "body": (
            "This circular is issued in supersession of the circular issued earlier on the same subject. "
            "The corrigendum is issued to correct a clerical error in the original notification. "
            "The figures mentioned in paragraph 4 may be read as figures for the preceding financial year. "
            "The organisation shall submit a utilisation certificate for the grant received. "
            "The contractor must furnish a performance guarantee in the prescribed form. "
            "The utilisation certificate is to be submitted within 60 days of the close of the financial year. "
            "The last date for the submission of the utilisation certificate is 30 June. "
            "The extension of time is granted up to 30 September. "
            "The period of the scheme has been extended by 6 months. "
            "The Authority has noted with satisfaction the progress made in the implementation of the Scheme. "
            "For record only. No acknowledgement is necessary."
        ),
    },
    {
        "id": "N5",
        "title": "Notice: Deficiency memo in a citizen application",
        "issuer": "Office of the Sub-Divisional Magistrate",
        "subject": "Rectification of deficiency in the application for issuance of certificate",
        "body": (
            "This office has received your application for the issuance of the certificate. "
            "On scrutiny of the application the following deficiency has been noticed. "
            "The applicant is required to submit a self-attested copy of the Aadhaar card along with the application. "
            "Applicants are required to attach the income certificate issued by the Tehsildar. "
            "The applicant shall submit the proof of residence such as a utility bill or rent agreement. "
            "The applicant shall submit an affidavit duly attested by a notary. "
            "The applicant shall rectify the defects within 7 days of the communication of the deficiency. "
            "The application should reach this office within 10 days of the issue of this notice. "
            "The applicant shall be given 15 days to file the reply. "
            "The application was forwarded to the sanctioning authority for consideration. "
            "The document is issued without any prejudice to the rights of the parties. "
            "You are requested to submit the pending documents at the earliest."
        ),
    },
]

# --------------------------------------------------------------------------
# Verified reference facts. Sources checked against official domains only.
# --------------------------------------------------------------------------
PORTALS = [
    {
        "name": "CPGRAMS (public grievance portal)",
        "url": "https://pgportal.gov.in/",
        "use": "Lodge a grievance against a central or state government department and track it.",
    },
    {
        "name": "RTI Online",
        "url": "https://rtionline.gov.in/",
        "use": "File a Right to Information request or first appeal with a central ministry or department.",
    },
    {
        "name": "eGazette of India",
        "url": "https://egazette.gov.in/",
        "use": "Check whether a notification you have received actually appears in the Official Gazette.",
    },
    {
        "name": "India Code",
        "url": "https://www.indiacode.nic.in/",
        "use": "Read the text of a central or state Act, and the rules and regulations made under it.",
    },
    {
        "name": "DigiLocker",
        "url": "https://www.digilocker.gov.in/",
        "use": "Pull an issued digital document instead of a photocopy when a notice asks for a copy.",
    },
    {
        "name": "UMANG",
        "url": "https://web.umang.gov.in/",
        "use": "Reach many central and state services from one place, including EPFO and pension services.",
    },
]

HELPLINES = [
    {
        "name": "National Consumer Helpline",
        "number": "1915",
        "alt": "1800-11-4000",
        "url": "https://consumerhelpline.gov.in/",
        "use": "Complaints about a good or a service, including a paid service that was not delivered.",
    },
    {
        "name": "CPGRAMS grievance helpdesk",
        "number": "via pgportal.gov.in",
        "alt": "",
        "url": "https://pgportal.gov.in/",
        "use": "Grievances against a government department, including delay in a service.",
    },
    {
        "name": "Cyber crime financial fraud helpline",
        "number": "1930",
        "alt": "",
        "url": "https://cybercrime.gov.in/",
        "use": "If a notice you received turns out to be a fraud and money has been lost.",
    },
]

# --------------------------------------------------------------------------
# Verified scheme facts used by the agent when a notice names a scheme.
# --------------------------------------------------------------------------
SCHEMES = [
    {
        "key": "pm-kisan",
        "name": "PM-KISAN Samman Nidhi",
        "matches": ["pm-kisan", "pmkisan", "kisan samman", "income support to landholding farmer"],
        "benefit": "Rs 6,000 per year in three equal instalments to eligible landholding farmer families.",
        "who": "Landholding farmer families with cultivable landholding in their names.",
        "portal": "https://pmkisan.gov.in/",
        "source": "pmkisan.gov.in (scheme portal)",
    },
    {
        "key": "epfo",
        "name": "Employees' Provident Fund",
        "matches": ["provident fund", "epfo", "epf", "uan", "employees' state insurance"],
        "benefit": "Accumulated provident fund balance, pension and insurance benefits for covered employees.",
        "who": "Employees in establishments covered by the Act, holding a Universal Account Number.",
        "portal": "https://unifiedportal-mem.epfindia.gov.in/memberinterface/",
        "source": "unifiedportal-mem.epfindia.gov.in (EPFO member portal)",
    },
    {
        "key": "aadhaar",
        "name": "Aadhaar (UIDAI)",
        "matches": ["aadhaar", "uidai", "e-kyc", "myaadhaar"],
        "benefit": "Identity, address update, and Aadhaar-based authentication services.",
        "who": "Residents enrolled with UIDAI.",
        "portal": "https://uidai.gov.in/",
        "source": "uidai.gov.in (UIDAI)",
    },
    {
        "key": "digilocker",
        "name": "DigiLocker",
        "matches": ["digilocker", "digital document", "digital locker"],
        "benefit": "Access to issued digital documents and certificates that can be shared and verified.",
        "who": "Any resident with a mobile number and an Aadhaar or other accepted credential.",
        "portal": "https://www.digilocker.gov.in/",
        "source": "digilocker.gov.in",
    },
    {
        "key": "parivahan",
        "name": "Parivahan Sewa",
        "matches": ["parivahan", "driving licence", "driving license", "learner", "vehicle registration"],
        "benefit": "Driving licence and vehicle registration services, including renewal and duplicate issue.",
        "who": "Licence holders and vehicle owners.",
        "portal": "https://parivahan.gov.in/",
        "source": "parivahan.gov.in",
    },
    {
        "key": "income-tax",
        "name": "Income Tax e-Filing",
        "matches": ["income tax", "return of income", "e-filing", "pan", "assessment year"],
        "benefit": "Filing of income tax returns and related forms, and e-PAN services.",
        "who": "Taxpayers and persons seeking a PAN.",
        "portal": "https://www.incometax.gov.in/",
        "source": "incometax.gov.in (Income Tax Department)",
    },
]


def scheme_for(text: str):
    """Return the first scheme whose trigger phrases appear in the text."""
    low = (text or "").lower()
    for s in SCHEMES:
        if any(m in low for m in s["matches"]):
            return s
    return None


def corpus_size() -> int:
    return len(SENTENCES)
