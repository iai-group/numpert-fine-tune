"""Apply pre-written abstractive passages to all AVeriTeC dev files in-place."""

import json
from pathlib import Path

DATA_ROOT = Path(__file__).parent.parent.parent / "data" / "averitec"

PASSAGES = {
    "averitec_dev_000001": (
        'Macron described Islam as a religion "in crisis" worldwide and vowed to present a bill '
        "in December to strengthen France's church-state separation law. Pakistan's Prime Minister "
        "Imran Khan criticised Macron, accusing him of encouraging Islamophobia by attacking Islam "
        "rather than terrorists and deliberately provoking Muslims by encouraging the display of "
        "blasphemous cartoons. The tweets attributed to the Pakistani Consulate in France originated "
        "from @PakConsulateFr, which is not the official Twitter account of the Pakistani Consulate "
        "in France. No evidence was found that France cancelled Pakistani citizens' visas or deported "
        "any Pakistanis in connection with Khan's comments."
    ),
    "averitec_dev_000002": (
        "Don Gaetz is the retired co-founder of VITAS Healthcare Corp., a hospice care provider, "
        "and also served as president of the National Hospice Organization. In 2004, VITAS was "
        "acquired by Chemed, the parent company of Roto-Rooter, for $400 million. A lawsuit against "
        "Chemed was filed in 2013—nine years after Don Gaetz sold the company—alleging fraud dating "
        "back to at least 2002. Don Gaetz was not a party to the lawsuit and denied that the matters "
        "in the complaint related to his work with Vitas. No evidence was found that Matt Gaetz "
        "personally worked for Chemed Corporation, Vitas Hospice Services, or Vitas Healthcare."
    ),
    "averitec_dev_000003": (
        "As of October 31, 2020, the United States had recorded 230,512 COVID-19 deaths. Research "
        "indicates that the delayed U.S. pandemic response was devastating: 117,858 Americans died "
        "in the four months following the first 15 confirmed cases, compared to only 8,863 in "
        "Germany over an equivalent period. Scaling Germany's population to the size of the U.S., "
        "approximately 70% of U.S. coronavirus deaths—around 161,358 of the 230,512 total—might "
        "have been prevented had the U.S. acted as effectively as Germany."
    ),
    "averitec_dev_000004": (
        "Donald Trump paid $750 in federal income taxes in 2017. During the 2016 presidential "
        "debate with Hillary Clinton, Trump boasted that not paying taxes made him smart."
    ),
    "averitec_dev_000005": (
        "Amy Coney Barrett was confirmed as a U.S. Supreme Court Justice."
    ),
    "averitec_dev_000006": (
        "No evidence was found online that Breitbart News had reported on the daughter of "
        "Delaware Democratic Senator Chris Coons and seven other underage girls being featured "
        "on Hunter Biden's laptop. The claim was made by Lauren Witzke (@LaurenWitzkeDE) on "
        "Twitter, who stated she had received information from sources close to the investigation."
    ),
    "averitec_dev_000007": (
        "Rock star Jon Bon Jovi performed a small concert at a drive-in rally in eastern "
        "Pennsylvania for Joe Biden, performing three songs and praising Biden's response to "
        "the coronavirus. About 230 cars with people were in attendance at the event."
    ),
    "averitec_dev_000008": (
        "Medical professionals recommend late cord clamping, performed approximately one to "
        "three minutes after birth, for all births."
    ),
    "averitec_dev_000009": (
        "According to the Johns Hopkins University COVID-19 dashboard, as of October 23, 2020, "
        "the U.S. had recorded 3,353,056 recoveries out of 8,406,718 cases, yielding a case "
        "recovery rate of 39.88%. Globally, there were 28,349,541 recoveries out of 41,705,699 "
        "cases, a recovery rate of 67.97%. In the U.S., 223,042 deaths were reported out of "
        "8,406,718 cases as of October 23, equivalent to a case fatality rate of 2.65%. The "
        "global case fatality rate stood at 2.72%."
    ),
    "averitec_dev_000010": (
        "The United States managed to cut its carbon dioxide output in 2019 under the Trump "
        "administration. However, the lowest U.S. carbon emissions on record were 6,373 metric "
        "tons of carbon dioxide, recorded in 1991 when George H. W. Bush was president. While "
        "carbon emissions did decline during the Trump presidency, that reduction did not reach "
        "the historic low previously recorded."
    ),
    "averitec_dev_000011": (
        "India's imports from China did not increase between April and August 2020; they "
        "actually decreased. Commerce and Industry Minister Piyush Goyal confirmed that "
        "India's imports from China declined by 27.63 per cent during April–August 2020-21 "
        "over the corresponding period of the previous year."
    ),
    "averitec_dev_000012": (
        "The claim was made on 20 October 2020. From 2019 onwards, there is no longer any "
        "penalty for not having health insurance in the United States."
    ),
    "averitec_dev_000013": (
        "The majority of deaths in the 1918–1919 influenza pandemic likely resulted directly "
        "from secondary bacterial pneumonia caused by common upper respiratory tract bacteria, "
        "not from the influenza virus acting alone. The study in question does not mention "
        "masks and made no reference to deaths caused by wearing masks."
    ),
    "averitec_dev_000014": (
        "Research published in The Journal of Infectious Diseases found that most victims of "
        "the 1918–1919 influenza pandemic succumbed to bacterial pneumonia following influenza "
        "virus infection, rather than to the virus acting alone. The pneumonia occurred when "
        "bacteria that normally inhabit the nose and throat invaded the lungs along a pathway "
        "created when the virus destroyed cells lining the bronchial tubes and lungs. NIAID "
        'Director Anthony Fauci stated: "The virus landed the first blow while bacteria '
        'delivered the knockout punch." Pathologist Jeffery Taubenberger examined lung tissue '
        "from 58 soldiers who died in 1918 and 1919, finding evidence ranging from primary "
        "viral pneumonia to severe secondary bacterial pneumonia. Medical experts agree that "
        "bacteria, not influenza viruses, were the greatest cause of death during the 1918 "
        "pandemic, and Dr Fauci's publication made no claim that masks caused flu deaths."
    ),
    "averitec_dev_000015": (
        "India's imports from China declined sharply in 2020. From January to February 2020, "
        "India's imports from China were as low as US$9.5 billion, reflecting a 12.4 per cent "
        "trade decline in the initial two months of COVID-19. Imports fell to a record low of "
        "USD 3.2 billion in both April and May 2020, coinciding with pandemic-induced lockdowns, "
        "before recovering to USD 4.8 billion in June and USD 5.6 billion in July—nearly back "
        "to the pre-lockdown level of USD 5.8 billion in March. For the first seven months of "
        "2020, India's imports from China totalled USD 32.2 billion, still down 24.7% "
        "year-on-year. Commerce and Industry Minister Piyush Goyal confirmed that imports from "
        "China declined by 27.63% during April–August 2020."
    ),
    "averitec_dev_000016": (
        "Between March and August 2020, the U.S. recorded 1,336,561 deaths, of which 150,541 "
        "(67%) were attributed directly to COVID-19. During the same period, 225,530 were "
        "'excess' deaths—deaths occurring beyond what would normally be expected. Of those "
        "225,530 excess deaths, 150,541 (approximately two-thirds) were attributed to COVID-19, "
        "leaving roughly one-third of excess deaths attributable to other causes."
    ),
    "averitec_dev_000017": ("Pennsylvania did not reject mail-in ballots."),
    "averitec_dev_000018": (
        "The Hyderabad Metropolitan Water Supply and Sewerage Board (HMWSSB) confirmed via its "
        "official Twitter account that its staff were distributing one kilogram of bleaching "
        "powder and demonstrating how to clean sumps, while also delivering chlorine tablets to "
        "people's doorsteps. This was done to undertake comprehensive sanitation work to prevent "
        "the spread of seasonal and communicable diseases caused by rainwater entering water "
        "storage tanks."
    ),
    "averitec_dev_000019": (
        "It is not possible to vote more than once in a U.S. election. Additionally, a ballot "
        "application is not the same as a vote ballot; they are distinct documents."
    ),
    "averitec_dev_000020": (
        "There is no evidence that wearing a mask is a contributing factor or direct cause of "
        "secondary bacterial pneumonia. The National Institute of Allergy and Infectious Diseases "
        "stated that the 2008 study does not mention masks and had nothing to do with masks; "
        "instead, the study analysed data from fatal 1918 infections and showed that the majority "
        "of cases were associated with secondary bacterial pneumonias."
    ),
    "averitec_dev_000021": (
        "During his first year in office, Trump set a record for the most federal appeals judges "
        "confirmed in a president's first year, successfully appointing 12—surpassing Obama's "
        "three, Bush's six, and Nixon's and Kennedy's eleven each. Trump's success at installing "
        "top-level judges faster than any previous president was aided by outside groups, "
        "Republican control of both the presidency and the Senate, and Senate Majority Leader "
        "Mitch McConnell, who blocked Obama nominees to create 17 appeals court vacancies and "
        "also held open a Supreme Court seat for Trump. Early signs suggest Trump's judges are "
        "shifting courts on issues including reproductive healthcare, voting rights, and emergency "
        "relief for executive actions."
    ),
    "averitec_dev_000022": (
        "Amy Coney Barrett's nomination to the Supreme Court was generally supported by "
        "Republicans, who sought to confirm her before the 2020 U.S. presidential election. "
        "Her nomination was confirmed by a Senate majority that represented 15 million fewer "
        "people than the minority party."
    ),
    "averitec_dev_000023": (
        "At $171,000, the net worth of a typical white family is nearly ten times greater than "
        "that of a typical Black family ($17,150) in 2016, according to data from the DCFPI. "
        "When measured by average rather than median wealth, white average wealth ($929,800) is "
        "6.7 times greater than Black average wealth ($138,100), though average figures are more "
        "heavily influenced by very wealthy families."
    ),
    "averitec_dev_000024": (
        "There is no evidence that Joe Biden proposed a nationwide property tax increase. Local "
        "and state governments collect property taxes, not the federal government. The post "
        "claiming Biden proposed a 2% federal property tax increase is fake."
    ),
    "averitec_dev_000025": (
        "The LNP cut Queensland Health jobs on September 11, 2012. Health Minister Lawrence "
        "Springborg confirmed that the total number of Queensland Health redundancies was 2,754, "
        "reduced from an initial target representing FTE savings of 4,142 positions; dollar "
        "savings equivalent to 1,004 FTE positions and 384 growth positions reduced the net "
        "figure. The initial budgetary target was a reduction of $326 million. The subsequent "
        "Labor government reversed those cuts."
    ),
    "averitec_dev_000026": (
        "Wind energy from about 5,100 turbines powered 42% of Iowa's net electricity generation "
        "as of May 21, 2020. The remaining 2% of generation came from hydropower, biomass, "
        "petroleum, and solar."
    ),
    "averitec_dev_000027": (
        "A CDC report released on October 9, 2020, found that the percentage of people testing "
        "positive among older age groups began to increase after it increased among people aged "
        "0–17 and 18–24, suggesting that younger people were passing the virus to older people. "
        "Household transmission studies showed that children were rarely the index case in "
        "COVID-19 outbreaks, but it is highly likely that children can transmit the SARS-CoV-2 "
        "virus."
    ),
    "averitec_dev_000028": (
        'The Chinese embassy in The Bahamas dismissed as "fake news" claims that China was '
        "seeking court approval to kill more than 20,000 coronavirus patients. Embassy counsellor "
        'Haigang Yin stated: "There is no government in the world, during this age, that could '
        'do something like that," and emphasised that the Chinese government was making all '
        "efforts to cure and contain the virus, including sending thousands of doctors and nurses "
        "to Hubei province."
    ),
    "averitec_dev_000029": (
        "The United States declared coronavirus a public health emergency in January 2020. From "
        "March to April 2020, more than 22.7 million Americans lost their jobs. By around July "
        "to October 2020, approximately 12,445,532 people remained unemployed."
    ),
    "averitec_dev_000030": (
        "A report from Moody's Analytics found that the U.S.–China trade war, which began in "
        "early 2018, cost 300,000 American jobs through September 2018, based on an economic "
        "simulation."
    ),
    "averitec_dev_000031": (
        "Zimbabwe received US$2.78 billion from the Africa Export Import Bank. The bank's total "
        "loan portfolio stood at US$15.2 billion at the end of June 2020. Zimbabwe's share "
        "therefore represents approximately 18.2% of the bank's total loan disbursements."
    ),
    "averitec_dev_000032": (
        "In the months leading up to October 2020, unemployment insurance claims in the U.S. "
        "reached 30 million since February 2020."
    ),
    "averitec_dev_000033": (
        "There have been deaths from COVID-19 among people below the age of 20."
    ),
    "averitec_dev_000034": (
        "Richard A. Rothschild filed a patent application related to COVID-19 testing in May "
        "2020. According to Espacenet, a platform developed by the European Patent Office, this "
        "is a continuation of US patent application 62/240783 from 2015. The 2015 application "
        "describes a method for processing and displaying biometric data in synchronisation with "
        "video data and makes no mention of coronavirus or COVID-19."
    ),
    "averitec_dev_000035": (
        "The cumulative number of deaths in 2020 was far greater than the cumulative number of "
        "deaths in any single year from 2015 to 2019, based on historical death rate data."
    ),
    "averitec_dev_000036": (
        "According to the World Poverty Clock, approximately 33% of Nigerians were living in "
        "extreme poverty in both 2019 and 2020."
    ),
    "averitec_dev_000037": (
        "During the Obama/Biden administration, a total of 1,927 acts of clemency were granted."
    ),
    "averitec_dev_000038": (
        "The claim was made on 5 October 2020. People in the UK were required to self-isolate "
        "if they tested positive for COVID-19 in 2020. However, those required to self-isolate "
        "were not entitled to receive a support payment of £500 per week."
    ),
    "averitec_dev_000039": (
        "The claim was made on 4 October 2020. Wisconsin Act 185, which makes numerous changes "
        "to state law in response to the COVID-19 outbreak, was passed and took effect on "
        "April 17, 2020."
    ),
    "averitec_dev_000040": (
        "The figure of £90 billion appears to refer to an unofficial estimate of the total UK "
        '"tax gap" calculated by Richard Murphy, who runs the website Tax Research UK. HMRC '
        "defines the tax gap as the difference between the amount of tax that should in theory "
        "be paid and what is actually paid. The £90 billion figure does not refer only to tax "
        "avoidance; it encompasses the total tax gap from all sources."
    ),
    "averitec_dev_000041": (
        "The original article claims Dr Ebeneezer is a medical doctor and herbal practitioner "
        "at the Amuzu Hospital. Research does show a link between the amount of sexual activity "
        "or ejaculation and prostate cancer risk. However, the study in question did not "
        "conclude or recommend that men have sex or ejaculate 21 times per month."
    ),
    "averitec_dev_000042": (
        "There is no record of Joe Biden making a remark about creating 550 million new jobs "
        "on October 2, 2020 or any other date. The claim appears to be a misrepresentation of "
        "Biden saying in Michigan that he wants to create 5 million new manufacturing and "
        "technology jobs. If Biden had said he would create 550 million new jobs, media outlets "
        "would likely have reported it, yet none did."
    ),
    "averitec_dev_000043": (
        "Two E-6B Mercury aircraft, which can serve as command centres and launch land-based "
        "intercontinental ballistic missiles, were deployed around the time of Trump's positive "
        "COVID-19 test. However, that was not their purpose at the time of the deployment."
    ),
    "averitec_dev_000044": (
        "Nigeria gained independence on 1 October 1960. At that time, the population was "
        "approximately 45.1 million, based on estimates from July 1, 1960."
    ),
    "averitec_dev_000045": (
        "Nigeria's population is estimated to be in the region of 200 million."
    ),
    "averitec_dev_000046": (
        "Nigeria's population in 2020 was approximately 206 million people."
    ),
    "averitec_dev_000047": (
        "Nigeria first experienced military rule in 1966, which lasted 13 years. A subsequent "
        "period of military rule began in 1983 and lasted 16 years, ending in 1999."
    ),
    "averitec_dev_000048": (
        "Nigeria gained independence on 1 October 1960, making the country 60 years old in "
        "2020. Between 1960 and 2020, Nigeria spent 29 years under military rule."
    ),
    "averitec_dev_000049": (
        "Countries define urban populations in different ways: nearly two-thirds use an "
        "administrative definition, often combined with additional criteria such as population "
        "size, density, or economic activity. Denmark and Iceland classify urban areas at "
        "200 inhabitants, while the Netherlands and Nigeria use a threshold of 20,000, Mali "
        "30,000, and Japan 50,000. Nigeria was granted political independence in 1960 as a "
        "sovereign state under a parliamentary system. According to World Bank and UN data, "
        "Nigeria's urban population in 1960 was 6.956 million."
    ),
    "averitec_dev_000050": (
        "Nigeria gained independence on 1 October 1960. Data from the World Bank and the UN's "
        "department of economic and social affairs shows Nigeria's population was 45.1 million "
        "in 1960, with that estimate specifically calculated for 1 July 1960—three months "
        "before independence day."
    ),
    "averitec_dev_000051": (
        "Nigeria became independent on 1 October 1960. According to the World Urbanization "
        "Prospects: The 2018 Revision, prepared by the UN's department of economic and social "
        "affairs, Nigeria's urban population in 1960 was estimated at 6.96 million."
    ),
    "averitec_dev_000052": (
        "Nigeria achieved independence from British colonial rule on October 1, 1960. According "
        "to the UN's department of economic and social affairs, Nigeria's population was "
        "45.1 million on 1 July 1960."
    ),
    "averitec_dev_000053": (
        "Nigeria became independent in 1960. Data from the World Bank and the UN's department "
        "of economic and social affairs shows Nigeria's population was 45.1 million in 1960."
    ),
    "averitec_dev_000054": (
        "Nigeria returned to democracy in 1999 after two long periods of military rule—1966 to "
        "1979, and 1983 to 1998—during which the military wielded executive, legislative, and "
        "judicial power. In total, the military held sway for twenty-nine years out of the "
        "initial forty years of the country's post-independence history."
    ),
    "averitec_dev_000055": (
        "Nigeria became independent on 1 October 1960. The country first experienced military "
        "rule from 1966 to 1979, and subsequently experienced a further period of military rule "
        "after 1979 as well."
    ),
    "averitec_dev_000056": (
        "According to the UN's department of economic and social affairs, Nigeria's total urban "
        "population in 2020 was estimated at 107.11 million, while 99.04 million lived in rural "
        "areas, giving a total population of 206.15 million. The urban population represented "
        "approximately 51.96% of the total population."
    ),
    "averitec_dev_000057": (
        "Nigeria's 2020 urban population is estimated at 107.11 million, with a rural population "
        "of 99.04 million, for a total estimated population of 206.15 million. Nigeria had a "
        "52% urbanisation rate as of 2020."
    ),
    "averitec_dev_000058": (
        "As of 2020, approximately 52% of Nigeria's population lives in urban areas."
    ),
    "averitec_dev_000059": (
        "Nigeria's population at the time of the claim was 206,139,587 million."
    ),
    "averitec_dev_000060": (
        "According to the UN's department of economic and social affairs, Nigeria's urban "
        "population in 2020 was estimated at 107.11 million, while the rural population was "
        "estimated at 99.04 million. The total population of Nigeria in 2020 was approximately "
        "206.15 million, with the rural population representing approximately 51.96% of the "
        "total."
    ),
    "averitec_dev_000061": (
        "Nigeria experienced military dictatorship from 1966 to 1999, with an interregnum from "
        "1979 to 1983. In total, this amounts to 29 years of military rule."
    ),
    "averitec_dev_000062": (
        "On 1 October 1960, Nigeria was granted full independence from Great Britain, receiving "
        "its Freedom Charter from Princess Alexandra of Kent. According to the United Nations, "
        "Nigeria's urban population in 1960 was estimated at 6.956 million."
    ),
    "averitec_dev_000063": (
        "Nigeria faces multiple challenges including terrorism from Boko Haram, economic crisis, "
        'and corruption. Despite President Buhari declaring Boko Haram "technically defeated" '
        "in December 2015, attacks have continued, and more than three million people in "
        "northeastern Nigeria have been displaced. Oil price declines and production disruptions "
        "have compounded economic difficulties. Nigeria's population exceeds 200 million, with "
        "the UN projecting the country will add 200 million more urban residents over the next "
        "four decades."
    ),
    "averitec_dev_000064": (
        "Nigeria became independent on 1 October 1960. At independence, Nigeria's total "
        "population was approximately 45,138,458, while the urban population at mid-year was "
        "6,956,000."
    ),
    "averitec_dev_000065": (
        "President Muhammadu Buhari's independence day speech included claims about Nigeria's "
        "historical and current population, the legacy of military rule, reduced government "
        "revenue, and oil prices in other countries. Nigeria's population in 1960 was "
        "45.1 million."
    ),
    "averitec_dev_000066": (
        "Data from the World Bank and the UN's department of economic and social affairs shows "
        "Nigeria's total population was 45.1 million in 1960, with that estimate specifically "
        "for 1 July 1960—three months before independence day. According to the UN's World "
        "Urbanization Prospects 2018 Revision, Nigeria's urban population in 1960 was estimated "
        "at 6.96 million—nearly 7 million, supporting the claim that the urban population at "
        "independence was approximately 7 million."
    ),
    "averitec_dev_000067": (
        "In England during COVID-19 restrictions, the maximum fine for repeatedly flouting "
        "face covering rules was £3,200."
    ),
    "averitec_dev_000068": (
        "Asteroid 2018VP1 is approximately 6.5 feet in size, poses no threat to Earth, and had "
        "a 0.41% chance of entering Earth's atmosphere; if it did, it would disintegrate due to "
        "its extremely small size. A systematic review and meta-analysis of COVID-19 infection "
        "fatality rate (IFR) literature found an IFR of 0.68% (range 0.53%–0.82%), varying "
        "from a low of 0.17% to a high of 1.7% depending on location."
    ),
    "averitec_dev_000069": (
        "There are no credible news reports supporting the claim. The donation is attributed to "
        "Lai Mohammed, Nigeria's Minister of Information and Culture. Neither Nigeria nor Lai "
        "Mohammed appears on the FEC list of donors to the Biden campaign. Donations to U.S. "
        "presidential campaigns from foreign countries or nationals are not permitted."
    ),
    "averitec_dev_000070": (
        "The Reserve Bank of India explained that the new ₹500 notes in the Mahatma Gandhi "
        "(New) Series differ from the previous series in colour, size, theme, location of "
        "security features, and design elements. The new note measures 66mm × 150mm, is stone "
        "grey in colour, and features the Red Fort as the predominant theme. None of the "
        "official features described by the RBI mention a green thread."
    ),
    "averitec_dev_000071": (
        "Accusations that Massachusetts committed voter fraud by destroying over 1 million "
        "ballots are false. It is not normal to destroy voting ballots after a U.S. election, "
        "and all ballots cast in Massachusetts still exist."
    ),
    "averitec_dev_000072": (
        "Global investment firm KKR (Kohlberg Kravis Roberts & Co. L.P.) proposed to invest "
        "Rs 5,550 crore in Reliance Retail Ventures Ltd (RRVL), a subsidiary of Reliance "
        "Industries."
    ),
    "averitec_dev_000073": (
        "New Zealand's Energy and Resources Minister Megan Woods stated that the proposed "
        "pumped hydro project at Lake Onslow could create thousands of jobs, make wholesale "
        "electricity cheaper in the long run, and decarbonise the grid by reducing reliance "
        "on coal and gas. The project would eliminate the risk premium in electricity prices "
        "that can drive up spot prices for wholesale electricity in dry years. The ICCC "
        "committee's report did not consider pumped hydro in its projections, and traditional "
        "hydro generation carries the risk that insufficient rainfall will cut generation and "
        "increase prices."
    ),
    "averitec_dev_000074": (
        "In the 2016 U.S. presidential election, more than 41% of all ballots were cast before "
        "Election Day. By early October 2020, 3.3 million people had already voted, compared "
        "to only 74,836 at a similar point before the 2016 election. University of Florida "
        "professor Michael McDonald estimated total 2020 general election turnout at "
        "150 million voters, meaning approximately 2.2% of the electorate had voted by early "
        "October. In reality, approximately 240 million people were eligible to vote in 2020 "
        "and roughly 66.1% submitted ballots, totalling about 158 million."
    ),
    "averitec_dev_000075": (
        "The family of Justine Damond received a settlement of US$20 million."
    ),
    "averitec_dev_000076": (
        "According to the Fluzone High-Dose package insert, in one study no deaths were "
        "reported within 28 days post-vaccination, while 23 deaths were reported during days "
        "29–180. In a second study, 167 deaths were reported within 6 to 8 months "
        "post-vaccination—83 (0.5%) among Fluzone High-Dose recipients and 84 (0.5%) among "
        "standard Fluzone recipients; 6 deaths occurred within 30 days, all among Fluzone "
        "High-Dose recipients. The package insert states these data do not provide evidence "
        "for a causal relationship between deaths and vaccination with Fluzone High-Dose. "
        "As of September 2020, Canada had recorded 9,249 COVID-19 deaths and the United States "
        "had recorded 209,498 COVID-19 deaths. Flu vaccines are considered the most effective "
        "way to prevent influenza illness and influenza-related complications, particularly "
        "important during the COVID-19 pandemic."
    ),
    "averitec_dev_000077": (
        "Resistbot is an automated service that converts texts into emails and sends them to "
        "elected representatives. Users can reach Resistbot by texting 50409. Texting "
        '"RBG" to 50409 adds the sender\'s name to an online petition, which is then '
        "generated as an email to elected representatives."
    ),
    "averitec_dev_000078": (
        "Zimbabwe's Ministry of Health and Child Care confirmed on 20 February 2020 that a "
        "reported suspected COVID-19 case had been ruled out as negative, and that Zimbabwe "
        "remained coronavirus free at that time."
    ),
    "averitec_dev_000079": (
        "President Clinton announced Ruth Bader Ginsburg as a Supreme Court nominee on "
        "June 15, 1993, and she was confirmed by the United States Senate on August 3, 1993. "
        "There was no U.S. election 43 days after her confirmation in August 1993."
    ),
    "averitec_dev_000080": (
        "Many people predicted that Donald Trump would contract COVID-19; John Cammo was not "
        "the only person to do so."
    ),
    "averitec_dev_000081": (
        "John Paul Stevens holds the record for the fewest days between Supreme Court "
        "nomination and confirmation at 19 days, followed by Amy Coney Barrett at 27 days. "
        "The overall average from nomination to final Senate vote is 68.2 days (median 69.0 "
        "days). Among the nine current Justices, the average is 72.1 days (median 73.0 days), "
        "with Amy Coney Barrett waiting the fewest days (27) and Clarence Thomas the greatest "
        "number (99)."
    ),
    "averitec_dev_000082": (
        "The 2011 Curry International Tuberculosis Center manual on tuberculosis infection "
        "control states that individuals with suspected or known infectious TB should wear a "
        "surgical mask when not in an airborne infection isolation room or a local exhaust "
        "ventilation enclosure, in order to block aerosols produced by coughing, talking, and "
        "breathing. Staff are to monitor patients for compliance and replace masks when damp. "
        "The Curry International Tuberculosis Center creates and disseminates resources and "
        "performs research to control and eliminate tuberculosis in the United States and "
        "internationally."
    ),
    "averitec_dev_000083": (
        "The Affordable Care Act (ACA) provides individuals and families access to affordable "
        "health insurance, preventing insurers from refusing coverage based on gender or "
        "pre-existing conditions, eliminating lifetime or annual limits, allowing young adults "
        "to stay on family plans until age 26, and offering Medicare prescription drug "
        "discounts. The Trump administration asked the Supreme Court to repeal the ACA in its "
        'entirety, stating "the entire ACA act must fall", consistent with Trump\'s 2016 '
        "campaign pledge and Republican lawmakers' longstanding opposition to the law since "
        "its 2010 inception."
    ),
    "averitec_dev_000084": (
        "Senator Gimenez was earning around $150,000 after voluntarily cutting his salary when "
        "he became mayor in 2011. He subsequently increased his salary by 67% because he had "
        "been taking a pay cut since 2011. He also increased his own pension."
    ),
    "averitec_dev_000085": (
        "Reed Hastings is the founder and was the CEO of Netflix. As of 2021, Netflix has "
        "two Co-Chief Executive Officers: Reed Hastings and Ted Sarandos."
    ),
    "averitec_dev_000086": (
        "No estates in Nigeria were marked for demolition in 2020 by the Abuja Metropolitan "
        "Management Council."
    ),
    "averitec_dev_000087": (
        "Each eligible U.S. citizen is allowed to cast only one vote during elections—a basic "
        "tenet of democracy. A voter who received multiple envelopes from the Center for Voter "
        'Information received "Vote At Home Ballot Request Forms," not actual ballots.'
    ),
    "averitec_dev_000088": (
        "Rafale aircraft were being inducted into service at the time of the claim. None of "
        "the inducted aircraft crashed."
    ),
    "averitec_dev_000089": (
        "The claim was made during a debate on October 14, 2020. On that date, Georgia's "
        "health department reported 7,470 total COVID-19 deaths and 334,601 COVID-19 cases, "
        "while North Carolina's health department reported 3,856 total COVID-19 deaths and "
        "236,407 confirmed cases. CMS reported 1,328 nursing home COVID-19 deaths in North "
        "Carolina and 1,897 nursing home COVID-19 deaths in Georgia."
    ),
    "averitec_dev_000090": (
        "The tweet in question did not come from Donald Trump's official Twitter account, "
        "@realDonaldTrump. Trump's official account made no mention of the officers shot and "
        "critically injured."
    ),
    "averitec_dev_000091": (
        "Under the 2015 Joint Comprehensive Plan of Action (JCPOA), Iran was restricted to "
        "a total enriched uranium stockpile of no more than 300 kg of up to 3.67% enriched "
        "uranium hexafluoride for 15 years, equivalent to 202.8 kg. Iran continued enrichment "
        "after the deal but did not enrich above 3.67% in 2016. By January 2020, a senior "
        "Iranian official claimed the stockpile had reached 1,200 kg, more than doubling in "
        "three months, while a separate report indicated a stockpile of around 2,105 kg of "
        "enriched uranium. The IAEA found approximately 550 kg of uranium hexafluoride enriched "
        "to less than 4.5% as of late 2019; if further enriched, this quantity could yield "
        "more than 30 kg of weapons-grade uranium, enough for one fission bomb."
    ),
    "averitec_dev_000092": (
        "The claim was made on 13 September 2020. According to data released by the Nigeria "
        "Bureau of Statistics, the unemployment rate in Edo State dropped from 25.1% in the "
        "third quarter of 2018 to 19% in the second quarter of 2020."
    ),
    "averitec_dev_000093": (
        "There is no evidence, nor any news reporting, to support the claim that Netflix CEO "
        "Reed Hastings was arrested for possession of child pornography."
    ),
    "averitec_dev_000094": (
        "The Indian Government set up five training centres between 2009 and 2010 to provide "
        "underprivileged communities—including Scheduled Tribes, Scheduled Castes, women, and "
        "minorities—with free residential coaching for civil services and other government "
        "exams. One of the five centres is Jamia Millia Islamia, which was granted minority "
        "institution status by the NCMEI and can reserve up to 50% of seats for Muslims; "
        "however, the institution serves all underprivileged communities, not Muslims only. "
        "The government provides additional free coaching opportunities beyond the five centres, "
        "and none of the five centres are Muslim-only institutions."
    ),
    "averitec_dev_000095": (
        "Section 230 of the 1996 Communications Decency Act protects online service providers "
        "from lawsuits based on user-generated content. A 49–46 Senate vote took place to "
        "confirm Nathan A. Simington of Virginia as a member of the Federal Communications "
        "Commission for a term of five years from July 1, 2019. The 2021 bill related to this "
        "vote contains no mention of Section 230."
    ),
    "averitec_dev_000096": (
        "No results confirm that Nita Ambani is giving Kangana Ranaut Rs 200 crore for the "
        "construction of a new studio."
    ),
    "averitec_dev_000097": (
        "Joe Biden's gross incomes above $400,000 since 1998 include: $407,009 in 2013, "
        "$11,031,309 in 2017, and $4,580,437 in 2018."
    ),
    "averitec_dev_000098": (
        "Between 20 August and 26 August 2020, 31,388 people were identified as having come "
        "into close contact with someone who tested positive for COVID-19 in the UK. Of these, "
        "69.4% were reached and asked to self-isolate."
    ),
    "averitec_dev_000099": (
        "Approximately 80 million mail ballots were expected to be cast in the 2020 U.S. "
        "election. This figure represents the estimated total number of all mail ballots, "
        "not unsolicited ones."
    ),
    "averitec_dev_000100": (
        "South African police stated that social media posts depicting a building on Johannes "
        "Ramokhoase Street in Pretoria's CBD as a human trafficking centre are fake news. "
        "Police confirmed the building is in fact a residential facility accommodating "
        "police officers."
    ),
    "averitec_dev_000101": (
        "A photo exists of an SUV handover ceremony that took place at Dr Alex Ekwueme Square "
        "in Awka. All beneficiaries attended the ceremony, including the Chairman of the "
        "Anambra State Traditional Rulers Council, Igwe Nnaemeka Achebe, who also serves as "
        "Chairman of the South East Traditional Rulers Council."
    ),
    "averitec_dev_000102": (
        'Donald Trump did not tweet "THE NEXT 30 TO 60 DAYS WILL BE THE BIGGEST DRAINING OF '
        "THE SWAMP EVER.\" The tweet was posted from a fake account impersonating Trump's real "
        "Twitter handle, originating from New Zealand."
    ),
    "averitec_dev_000103": (
        "The Uyghur population is concentrated in the Xinjiang Uyghur Autonomous Region of "
        "northwest China, with approximately 10 million Uyghurs in China and at least 300,000 "
        "in Central Asian republics. The birthrate for Uyghurs in Xinjiang dropped from 14.85 "
        "per 1,000 in 2010 to 10.69 per 1,000 in 2018. In the predominantly Uyghur prefectures "
        "of Kashgar and Hotan, yearly birthrates between 2010 and 2018 ranged from 24.27 to "
        "8.17 per 1,000, with an overall average of approximately 16+ and a 2018 figure of "
        "8.17. Birth rates in Hotan and Kashgar plunged more than 60% from 2015 to 2018, "
        "compared to a 24% drop across Xinjiang and just 4.2% nationwide. The Uyghur birth "
        "rate of about 12 per 1,000 does not match official data from Xinjiang Statistical "
        "Yearbooks, and the 2018 figure for Hotan and Kashgar was below both regional and "
        "national averages. Researchers attribute the dramatic decline to a birth control "
        "campaign involving sterilisations, IUDs, and mandatory birth control measures."
    ),
    "averitec_dev_000104": (
        "President Trump issued a memorandum ordering Treasury Secretary Steven Mnuchin to "
        "suspend Social Security payroll taxes for some employers and employees from "
        "September 1, 2020 until December 31, 2020. While the tax cut could potentially be "
        "made permanent, the White House stated it would have zero impact on Social Security, "
        "as an automatic contribution from the general fund would cover Social Security "
        "funding."
    ),
    "averitec_dev_000105": (
        "About 460,000 people attended the Sturgis Motorcycle Rally. South Dakota confirmed "
        "124 COVID-19 cases tied to the rally as of September 8, and at least 290 people in "
        "12 other states tested positive after attending. Based on 414 total cases out of "
        "460,000 attendees, the infection rate was 0.09%, not 0.02%."
    ),
    "averitec_dev_000106": (
        "South Dakota reported 124 COVID-19 cases among state residents who attended the "
        "Sturgis Rally. Combined with cases from other states, a total of 414 cases were "
        "linked to the rally out of approximately 460,000 attendees, yielding an infection "
        "rate of 0.09%."
    ),
    "averitec_dev_000107": (
        "The legal age of consent in California is 18 years old, the age at which a person "
        "can legally marry and engage in sexual acts. For consensual sexual relations between "
        "a teenager aged 15 or older and a partner within 10 years of age, California law "
        "already granted judges discretion over sex offender registration for vaginal "
        "intercourse but mandated registration for oral and anal intercourse. Senator Scott "
        "Wiener introduced SB 145, which extends that same judicial discretion to other forms "
        "of intercourse, ending what the bill's proponents describe as discrimination against "
        "LGBT young people. SB 145 does not change whether particular behaviour is a crime or "
        "alter potential sentencing."
    ),
    "averitec_dev_000108": (
        "The United States and India report GDP growth and contraction differently. The U.S. "
        "GDP dropped by 9.1% during April to June 2020 on an annualised basis, while India's "
        "GDP contracted by 23.9%. When compared using the same reporting methodology, the "
        "U.S. GDP contraction during April to June 2020 was 9.1%."
    ),
    "averitec_dev_000109": (
        'A Tech Nation "Jobs & Skills Report 2020" analysed job vacancy advertising in the '
        "UK from June to August 2020 using Adzuna data, and a Times article based on it stated "
        "that in Belfast and Cambridge the tech sector employs 26% of the workforce. However, "
        "using BRES data, the two relevant industrial sectors—information and communication, "
        "and professional, scientific, and technical—employed an estimated 32,927 people in "
        "Belfast, representing 14% of the city's workforce. Counting only IT and engineering "
        "roles of the type advertised on Adzuna, an estimated 18,041 people are employed in "
        "such jobs in Belfast, representing 7.8% of Belfast's jobs."
    ),
    "averitec_dev_000110": (
        "In the 2018/2019 year, 270 cases of torture and 3,835 cases of assault were brought "
        "against the South African police department."
    ),
    "averitec_dev_000111": (
        "The South African police force employed 132,310 people in 2002/03 and 192,277 in "
        "2018/19, representing a growth of 45.3% over that period."
    ),
    "averitec_dev_000112": (
        "Police violence has increased in South Africa. In 2018, South African police killed "
        "558 people, and 201 additional people died in custody at the hands of South African "
        "police."
    ),
    "averitec_dev_000113": (
        "The North American bison was nearly brought to extinction, though it was not reduced "
        "to only 27 individuals by colonisers. Through ongoing conservation work, bison have "
        "recovered from a low of fewer than 1,000 individuals; today, 19 herds in 12 states "
        "roam the American Plains, and about 500,000 bison live on preserves and ranches "
        "where they are raised for their meat."
    ),
    "averitec_dev_000114": (
        "GiveSendGo, a Christian crowdfunding website, set up a fundraising account for Kyle "
        "Rittenhouse's legal defence. The site initially raised over $80,000 and subsequently "
        "reported raising more than $100,000 for Rittenhouse's defence."
    ),
    "averitec_dev_000115": (
        "When Trump began campaigning for president, the southwest border had 654 miles of "
        "primary barriers. Under Trump, that increased by three miles to 657 miles. The Trump "
        "administration mainly replaced barriers installed by previous administrations with new "
        "ones expected to be more effective. Mexico did not pay for any construction; American "
        "taxpayers covered the costs."
    ),
    "averitec_dev_000116": (
        "Joe Biden stated he would raise taxes for anyone making over $400,000 and impose no "
        "new taxes on anyone making under $400,000. Biden would also raise taxes on corporations "
        "by increasing the corporate income tax rate and imposing a corporate minimum book tax."
    ),
    "averitec_dev_000117": (
        "Security votes are monthly budgetary allocations provided by Nigeria's federal "
        "government to the 36 states for financing security, disbursed at officials' discretion "
        "and not subject to legislative oversight or independent audit. Transparency "
        "International estimates these secretive cash expenditures add up to over $670 million "
        "(N241.2 billion) annually. In 2020, the Edo State government received 6.5 billion "
        "naira through security voting, equating to approximately 417 million naira per month."
    ),
    "averitec_dev_000118": (
        "According to eyewitness videos and the criminal complaint, Kyle Rittenhouse shot and "
        "killed Joseph Rosenbaum in a parking lot after Rosenbaum chased him and tried to grab "
        "his gun. Rittenhouse then began running down the street, tripped and fell, and was "
        "approached by Anthony Huber carrying a skateboard; Rittenhouse fired one round that "
        "killed Huber. He then fired a shot that hit Gaige Grosskreutz in the arm. Rittenhouse "
        "subsequently approached police at the scene. Kenosha Police Chief Daniel Miskinis "
        "defended his officers' actions, stating they did not see Rittenhouse as a threat and "
        "could not hear crowd yelling over other noise, as they were focused on injuries "
        "further down the street."
    ),
    "averitec_dev_000119": (
        "The first night of the 2020 Republican National Convention on Monday, August 24 was "
        "broadcast on cable and television networks. According to Nielsen, which statistically "
        "estimates the number of Americans watching TV, an estimated 17 million people tuned "
        "in to watch the first night of the convention."
    ),
}


def patch_file(path: Path, id_to_doc: dict[str, str]) -> int:
    with open(path) as f:
        records = json.load(f)
    updated = 0
    for r in records:
        cid = r.get("claim_id", "")
        if cid in id_to_doc:
            r["doc"] = id_to_doc[cid]
            updated += 1
    with open(path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return updated


def main() -> None:
    print(f"Passages defined: {len(PASSAGES)}")

    # Patch dev numerical file
    updated = patch_file(DATA_ROOT / "averitec_dev_numerical.json", PASSAGES)
    print(f"averitec_dev_numerical.json: {updated} records updated")

    # Patch all perturbed dev_split files
    for p in sorted((DATA_ROOT / "perturbed" / "dev_split").glob("*.json")):
        n = patch_file(p, PASSAGES)
        print(f"  {p.name}: {n} records updated")

    print("Done.")


if __name__ == "__main__":
    main()
