import streamlit as st
import sqlite3, os, uuid
from datetime import datetime

def go_to(page):
    st.session_state.navigation = page
import base64

def render_img(img_val, height='160px'):
    if img_val and os.path.isfile(img_val):
        ext = os.path.splitext(img_val)[1].lower().lstrip('.') or 'png'
        if ext == 'jpg':
            ext = 'jpeg'
        with open(img_val, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/{ext};base64,{b64}" style="width:100%;height:{height};object-fit:cover;border-radius:12px;margin-bottom:10px">'
    return f'<div style="height:{height};border-radius:12px;background:radial-gradient(circle,#183148,#0b111b);display:flex;align-items:center;justify-content:center;font-size:3.5rem;margin-bottom:10px">{img_val or "♻️"}</div>'
import numpy as np
from PIL import Image, ImageStat, ImageFilter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor

@st.cache_resource
def train_carbon_model():
    X = np.array([
        [0,3],[0,2],[0,1],[0,0],
        [1,3],[1,2],[1,1],[1,0],
        [2,3],[2,2],[2,1],[2,0],
        [3,3],[3,2],[3,1],[3,0],
        [4,3],[4,2],[4,1],[4,0]
    ])
    y = np.array([
        18.5, 15.2, 11.8, 8.0,
        4.2,  3.5,  2.8,  1.9,
        6.5,  5.2,  4.0,  2.7,
        1.8,  1.4,  1.1,  0.7,
        5.0,  4.0,  3.0,  2.0
    ])
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model

def predict_eco_impact(category, condition):
    cats = {'Electronics': 0, 'Books': 1, 'College essentials': 2, 'Stationery': 3, 'Other': 4}
    conds = {'Fair': 0, 'Good': 1, 'Very good': 2, 'Excellent': 3}
    model = train_carbon_model()
    co2 = float(model.predict([[cats.get(category, 4), conds.get(condition, 1)]])[0])
    credits = int(round(co2 * 12))
    return round(co2, 2), credits

def analyze_item_condition(uploaded_file=None, condition_notes=""):
    score = 75.0
    signals = []

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB').resize((256, 256))
        crop = img.crop((38, 38, 218, 218))
        arr = np.asarray(crop, dtype=np.float32) / 255.0

        hsv = np.asarray(crop.convert('HSV'), dtype=np.float32) / 255.0
        sat_mean = float(hsv[:, :, 1].mean())
        val_std = float(hsv[:, :, 2].std())

        dark_spots = float((arr.mean(axis=2) < 0.18).mean())

        visual_score = 88.0 - (val_std * 55.0) - (dark_spots * 45.0) + (sat_mean * 12.0)
        score = max(35.0, min(98.0, visual_score))
        signals.append(f"Surface Uniformity: `{round((1 - val_std) * 100, 1)}%`")
        signals.append(f"Blemish/Wear Ratio: `{round(dark_spots * 100, 1)}%`")

    if condition_notes.strip():
        txt = condition_notes.lower()
        positive = ['new', 'mint', 'unused', 'unopened', 'perfect', 'scratchless', 'clean', 'like new', 'sealed']
        negative = ['scratch', 'dent', 'tear', 'torn', 'broken', 'crack', 'stain', 'faded', 'worn', 'missing', 'damage', 'used', 'highlighted']
        pos_hits = sum(1 for w in positive if w in txt)
        neg_hits = sum(1 for w in negative if w in txt)
        score = max(25.0, min(99.0, score + (pos_hits * 8.0) - (neg_hits * 12.0)))
        signals.append(f"Notes Sentiment: `+{pos_hits} positive / -{neg_hits} wear cues`")

    if score >= 82:
        cond = 'Excellent'
    elif score >= 68:
        cond = 'Very good'
    elif score >= 52:
        cond = 'Good'
    else:
        cond = 'Fair'

    return cond, round(score, 1), " · ".join(signals) if signals else "Baseline estimate"

def get_ai_swap_matches(query_text, items):
    if not query_text.strip() or not items:
        return []
    corpus = [query_text] + [f"{it['title']} {it['category']} {it['description']}" for it in items]
    vec = TfidfVectorizer(stop_words='english')
    tfidf = vec.fit_transform(corpus)
    scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    ranked = sorted(zip(scores, items), key=lambda x: x[0], reverse=True)
    return [(round(float(s) * 100, 1), it) for s, it in ranked if s > 0.05][:3]
BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, 'ecoswap.db')
UPLOADS = os.path.join(BASE, 'uploads')
os.makedirs(UPLOADS, exist_ok=True)

st.set_page_config(page_title='EcoSwap — Campus Exchange', page_icon='♻️', layout='wide', initial_sidebar_state='collapsed')

CSS = r'''<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Orbitron:wght@500;600;700;800&display=swap');
:root{--cyan:#4df8ff;--pink:#ff4fd8;--lime:#c7ff56;--bg:#070910;--muted:#91a4ba}
html,body,[data-testid="stApp"]{background:var(--bg)!important;color:#eff8ff;font-family:'Space Grotesk',sans-serif}
[data-testid="stApp"]{background:radial-gradient(ellipse at 15% 0%,#11263a 0,transparent 36%),radial-gradient(ellipse at 90% 12%,#29122c 0,transparent 32%),#070910!important}
.block-container{max-width:1450px;padding-top:1.1rem;padding-bottom:3rem}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:#0b101a!important;border-right:1px solid #39e9ff22}
.stButton button{border-radius:12px!important;border:1px solid #47f6ff80!important;background:linear-gradient(110deg,#00eaff20,#ff4fd820)!important;color:#f4fdff!important;font-weight:700!important;transition:all .2s ease!important}
.stButton button:hover{transform:translateY(-2px)!important;border-color:var(--pink)!important;box-shadow:0 0 22px #ff4fd833!important}
.stTextInput input,.stTextArea textarea,.stSelectbox div[data-baseweb="select"],.stNumberInput input{background:#0c1421!important;color:#f5faff!important;border:1px solid #4df8ff35!important;border-radius:10px!important}
.stTabs [data-baseweb="tab"]{color:#b5c7d9!important}.stTabs [aria-selected="true"]{color:var(--cyan)!important;border-bottom-color:var(--cyan)!important}
.neon-title{font-family:Orbitron,sans-serif;font-size:clamp(2.8rem,7vw,6.2rem);font-weight:800;letter-spacing:-.055em;line-height:.98;color:#f7fbff;text-shadow:0 0 12px #4df8ff35,0 0 44px #ff4fd820}
.grad{background:linear-gradient(90deg,var(--cyan),#fff 48%,var(--pink));-webkit-background-clip:text;background-clip:text;color:transparent}
.kicker{color:var(--cyan);font-size:.74rem;letter-spacing:.24em;font-weight:700;text-transform:uppercase}
.hero{position:relative;overflow:hidden;border:1px solid #49f5ff35;border-radius:28px;padding:clamp(25px,5vw,60px);background:linear-gradient(125deg,#101d2cdd,#151020e8);box-shadow:0 0 55px #00dfff0c,inset 0 0 50px #ff4fd80a;min-height:430px}
.hero:before{content:'';position:absolute;inset:0;background:linear-gradient(transparent 95%,#4df8ff12 96%),linear-gradient(90deg,transparent 95%,#4df8ff12 96%);background-size:34px 34px;opacity:.4;mask-image:linear-gradient(90deg,black,transparent 90%);pointer-events:none}
.orb{position:absolute;width:280px;height:280px;border-radius:50%;right:5%;top:13%;background:radial-gradient(circle,#4df8ff22,transparent 68%);border:1px solid #4df8ff30;box-shadow:0 0 50px #4df8ff12,0 0 0 28px #ff4fd80b,0 0 0 68px #4df8ff07;animation:float 7s ease-in-out infinite;pointer-events:none}
@keyframes float{0%,100%{transform:translateY(0) rotate(0)}50%{transform:translateY(-18px) rotate(6deg)}}
@keyframes pulseGlow{0%,100%{box-shadow:0 0 15px #4df8ff15}50%{box-shadow:0 0 30px #ff4fd822}}
@keyframes slideIn{from{opacity:0;transform:translateY(15px)}to{opacity:1;transform:translateY(0)}}
.hero-content{position:relative;z-index:1;animation:slideIn .65s ease both}
.metric{border:1px solid #ffffff14;border-radius:17px;padding:18px 20px;background:linear-gradient(145deg,#111c2a,#0c1019);animation:pulseGlow 5s infinite;min-height:112px}
.metric-label{font-size:.82rem;color:#9db0c6}.metric-value{font-family:Orbitron,sans-serif;color:var(--cyan);font-size:1.75rem;text-shadow:0 0 16px #4df8ff35}
.item{height:100%;border:1px solid #ffffff18;border-radius:19px;padding:17px;background:linear-gradient(145deg,#121c2b,#0b1019);transition:transform .22s,border .22s,box-shadow .22s;animation:slideIn .4s ease both}
.item:hover{transform:translateY(-6px);border-color:#ff4fd877;box-shadow:0 0 24px #ff4fd817}
.tag{display:inline-block;border:1px solid #4df8ff45;border-radius:99px;padding:3px 9px;color:var(--cyan);font-size:.72rem;background:#4df8ff0b}
.item-name{font-weight:700;font-size:1.06rem;margin:.45rem 0;color:#fff}.muted{color:var(--muted);font-size:.84rem}.section-title{font-family:Orbitron,sans-serif;font-size:1.55rem}
hr{border-color:#ffffff12!important}.footer{color:#60738b;text-align:center;padding:2rem 0;letter-spacing:.15em;font-size:.72rem}
</style>'''
st.markdown(CSS, unsafe_allow_html=True)

# database
def db():
    c=sqlite3.connect(DB, check_same_thread=False); c.row_factory=sqlite3.Row
    c.execute('''CREATE TABLE IF NOT EXISTS listings(id INTEGER PRIMARY KEY AUTOINCREMENT,title TEXT NOT NULL,category TEXT,description TEXT,condition TEXT,owner TEXT,exchange TEXT,status TEXT DEFAULT 'Available',image TEXT,created TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS requests(id INTEGER PRIMARY KEY AUTOINCREMENT,listing_id INTEGER,requester TEXT,message TEXT,status TEXT DEFAULT 'Pending',created TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT,request_id INTEGER,sender TEXT,message TEXT,created TEXT)''')
    if c.execute('SELECT COUNT(*) FROM listings').fetchone()[0]==0:
        seeds=[('Discrete Mathematics Notes','Books','Unit-wise handwritten notes with solved examples.','Good','Aarav','Free exchange','📚'),('Scientific Calculator','Electronics','All functions working. Looking for a simple swap.','Very good','Mira','Swap','🧮'),('Lab Coat · Size M','College essentials','Clean and barely used lab coat.','Excellent','Riya','Free','🥼'),('Drawing Instrument Kit','Stationery','Complete kit with compass, scale and pencils.','Good','Kabir','Swap','📐'),('Python Programming','Books','Beginner book with a few highlighted pages.','Good','Nisha','Free exchange','🐍'),('USB-C Hub','Electronics','HDMI and USB ports. Works perfectly.','Excellent','Dev','Swap','🔌')]
        c.executemany('INSERT INTO listings(title,category,description,condition,owner,exchange,status,image,created) VALUES(?,?,?,?,?,?,?, ?,?)',[(a,b,d,e,f,g,'Available',h,datetime.now().strftime('%Y-%m-%d')) for a,b,d,e,f,g,h in seeds])
    c.commit();return c
conn=db()

# Header / session name
st.markdown('<div style="display:flex;align-items:center;gap:10px"><span style="font-size:1.6rem">♻️</span><span style="font-family:Orbitron;font-weight:700;letter-spacing:.1em">ECO<span style="color:#4df8ff">SWAP</span></span><span class="tag">CAMPUS CIRCULARITY</span></div>',unsafe_allow_html=True)
with st.container():
    top1,top2=st.columns([4,1])
    with top1: st.caption('A student-powered exchange ecosystem')
    with top2: student=st.text_input('Your display name',value=st.session_state.get('student','Student'),label_visibility='collapsed',placeholder='Your name')
    st.session_state.student=student.strip() or 'Student'
if 'navigation' not in st.session_state: st.session_state.navigation='Home'
nav=st.radio('Navigation',['Home','Dashboard','Marketplace','Create listing','Exchange hub','About'],horizontal=True,label_visibility='collapsed',key='navigation')
st.markdown('<hr>',unsafe_allow_html=True)
listings=conn.execute('SELECT * FROM listings ORDER BY id DESC').fetchall(); available=[x for x in listings if x['status']=='Available']; reqs=conn.execute('SELECT requests.*,listings.title,listings.owner FROM requests JOIN listings ON listings.id=requests.listing_id ORDER BY requests.id DESC').fetchall()
exchanged=conn.execute("SELECT COUNT(*) FROM listings WHERE status='Exchanged'").fetchone()[0]

if nav=='Home':
    st.markdown(f'''<div class="hero"><div class="orb"></div><div class="hero-content"><div class="kicker">THE CAMPUS CIRCULAR ECONOMY</div><br><div class="neon-title">Swap smarter.<br><span class="grad">Waste less.</span></div><p style="max-width:650px;color:#a9bfd3;font-size:1.05rem;margin-top:22px">Your campus is full of things someone needs. Give good stuff another life, discover your next find, and make sustainability feel effortless.</p></div></div>''',unsafe_allow_html=True)
    st.write('')
    a,b,c=st.columns([1,1,1]);
    with a: st.markdown('<div class="metric"><div class="metric-label">Items waiting for a new life</div><div class="metric-value">'+str(len(available))+'</div></div>',unsafe_allow_html=True)
    with b: st.markdown('<div class="metric"><div class="metric-label">Exchanges completed</div><div class="metric-value">'+str(exchanged)+'</div></div>',unsafe_allow_html=True)
    with c: st.markdown('<div class="metric"><div class="metric-label">Swap requests in motion</div><div class="metric-value">'+str(len(reqs))+'</div></div>',unsafe_allow_html=True)
    st.write('')
    st.markdown('<div class="section-title">Your next great find <span class="grad">is out there.</span></div>',unsafe_allow_html=True)
    st.caption('Browse the latest items shared by your campus community.')
    cols=st.columns(3)
    for i,item in enumerate(available[:3]):
        with cols[i]:
            image_html = render_img(item["image"], "160px")
            st.markdown(f'<div class="item">{image_html}<span class="tag">{item["category"]}</span><div class="item-name">{item["title"]}</div><div class="muted">{item["condition"]} · {item["owner"]}</div><p>{item["description"][:105]}</p><b style="color:#ff4fd8">{item["exchange"]}</b></div>',unsafe_allow_html=True)
    st.write('')
    x, y = st.columns(2)
    with x:
        st.button('Explore marketplace →', use_container_width=True, on_click=go_to, args=('Marketplace',))
    with y:
        st.button('List an item +', use_container_width=True, on_click=go_to, args=('Create listing',))

elif nav=='Dashboard':
    st.markdown('<div class="kicker">YOUR IMPACT CONSOLE</div><div class="section-title">Campus <span class="grad">dashboard</span></div>',unsafe_allow_html=True)
    total=len(listings); pending=len([q for q in reqs if q['status']=='Pending']); yours=len([q for q in listings if q['owner']==student]); yourreq=len([q for q in reqs if q['requester']==student])
    cards=st.columns(4)
    for col,label,value in zip(cards,['Active listings','Your listings','Requests pending','Your requests'],[len(available),yours,pending,yourreq]):
        col.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',unsafe_allow_html=True)
    st.write('')
    st.markdown('#### Recent activity')
    if reqs:
        for q in reqs[:8]: st.markdown(f'<div class="item" style="margin-bottom:9px"><span class="tag">{q["status"]}</span> <b>{q["requester"]}</b> requested <b>{q["title"]}</b><div class="muted">{q["created"]}</div></div>',unsafe_allow_html=True)
    else: st.info('Activity will appear here once exchanges begin.')
    st.markdown('#### Campus impact')
    st.markdown(f'<div class="metric"><div class="metric-label">Items kept in circulation</div><div class="metric-value">{exchanged}</div><div class="muted">Every exchange is one less item sitting unused.</div></div>',unsafe_allow_html=True)

elif nav=='Marketplace':
    st.markdown('<div class="kicker">DISCOVER · BROWSE · REUSE</div><div class="section-title">The <span class="grad">marketplace</span></div>',unsafe_allow_html=True)
    q1,q2,q3=st.columns([2,1,1]); search=q1.text_input('Search items',placeholder='Search books, calculator, lab coat…'); cat=q2.selectbox('Category',['All','Books','Electronics','College essentials','Stationery','Other']); order=q3.selectbox('Sort',['Newest','A–Z'])
    filtered=[it for it in available if (not search or search.lower() in (it['title']+' '+it['description']+' '+it['category']).lower()) and (cat=='All' or it['category']==cat)]
    if order=='A–Z': filtered=sorted(filtered,key=lambda it:it['title'].lower())
    st.caption(f'{len(filtered)} items found')
    with st.expander('✨ AI Smart Swap Matchmaker (NLP Cosine Similarity)', expanded=False):
        ai_need = st.text_input('Describe what you need or want to swap for:', placeholder='e.g. engineering math textbook or scientific calculator')
        if ai_need:
            matches = get_ai_swap_matches(ai_need, available)
            if matches:
                for score, m in matches:
                    st.success(f"**{score}% Match** — **{m['title']}** ({m['category']} · {m['condition']}) by *{m['owner']}*")
            else:
                st.info('No close semantic matches yet—try broader keywords.')
    if not filtered: st.warning('No items match that search. Try another keyword or category.')
    for start in range(0,len(filtered),3):
        cols=st.columns(3)
        for col,it in zip(cols,filtered[start:start+3]):
            with col:
                img_html = render_img(it["image"], "160px")
                st.markdown(f'<div class="item">{img_html}<span class="tag">{it["category"]}</span><div class="item-name">{it["title"]}</div><div class="muted">Condition: {it["condition"]} · Owner: {it["owner"]}</div><p>{it["description"]}</p><b style="color:#ff4fd8">{it["exchange"]}</b></div>',unsafe_allow_html=True)
                co2_saved, eco_pts = predict_eco_impact(it['category'], it['condition'])
                st.caption(f"🌱 AI Impact Estimate: **-{co2_saved} kg CO₂e** · **+{eco_pts} Eco-Credits**")
                with st.expander('Request this item'):
                    msg=st.text_area('Message to owner',key=f'msg_{it["id"]}',placeholder='Hi! I am interested in this item…')
                    if st.button('Send swap request',key=f'request_{it["id"]}'):
                        if student==it['owner']: st.warning('You cannot request your own listing.')
                        else:
                            conn.execute('INSERT INTO requests(listing_id,requester,message,created) VALUES(?,?,?,?)',(it['id'],student,msg,datetime.now().strftime('%Y-%m-%d %H:%M')));conn.commit();st.success('Request saved!');st.rerun()

elif nav=='Create listing':
    st.markdown('<div class="kicker">PASS IT FORWARD</div><div class="section-title">Create a <span class="grad">listing</span></div>',unsafe_allow_html=True)
    with st.expander('🤖 AI Item Condition & CO₂ Impact Grader', expanded=True):
        v1, v2 = st.columns(2)
        preview_cat = v1.selectbox('Item Category', ['Books','Electronics','College essentials','Stationery','Other'], key='ai_cat')
        item_notes = v2.text_input('Describe item wear/usage', placeholder='e.g. Like new, no scratches / slightly worn cover', key='ai_notes')
        ai_photo = st.file_uploader('Upload item photo for AI physical condition check', type=['png','jpg','jpeg','webp'], key='ai_vision')
        if ai_photo or item_notes.strip():
            det_cond, wear_score, details = analyze_item_condition(ai_photo, item_notes)
            st.info(f"🔍 **AI Item Condition Grade:** **{det_cond}** (Health Score: `{wear_score}/100` — {details})")
        else:
            det_cond = 'Very good'
        p_co2, p_pts = predict_eco_impact(preview_cat, det_cond)
        st.success(f"🌍 **Predicted Sustainability Impact ({det_cond} condition):** Saves **{p_co2} kg CO₂e** and awards **{p_pts} Eco-Credits**")
    with st.form('create_listing',clear_on_submit=True):
        l1,l2=st.columns(2); title=l1.text_input('Item title*',placeholder='e.g. Data Structures textbook');category=l2.selectbox('Category',['Books','Electronics','College essentials','Stationery','Other']);desc=st.text_area('Description',placeholder='Condition details, what is included, pickup details…');l3,l4=st.columns(2);condition=l3.selectbox('Condition',['Excellent','Very good','Good','Fair']);exchange=l4.text_input('Exchange preference',placeholder='Free / Swap / ₹200 / Negotiable');photo=st.file_uploader('Item photo (optional)',type=['png','jpg','jpeg','webp']);submit=st.form_submit_button('Publish listing ✦')
    if submit:
        if not title.strip(): st.error('Please enter an item title.')
        else:
            image=''
            if photo:
                ext=os.path.splitext(photo.name)[1].lower(); path=os.path.join(UPLOADS,f'{uuid.uuid4().hex}{ext}');open(path,'wb').write(photo.getbuffer());image=path
            conn.execute('INSERT INTO listings(title,category,description,condition,owner,exchange,status,image,created) VALUES(?,?,?,?,?,?,?,?,?)',(title.strip(),category,desc.strip(),condition,student,exchange.strip() or 'Swap','Available',image,datetime.now().strftime('%Y-%m-%d')));conn.commit();st.success('Your listing is live!');st.balloons()

elif nav=='Exchange hub':
    st.markdown('<div class="kicker">REQUESTS & EXCHANGES</div><div class="section-title">Exchange <span class="grad">hub</span></div>',unsafe_allow_html=True)
    tab1,tab2=st.tabs(['Incoming requests','My requests'])
    with tab1:
        incoming=[r for r in reqs if r['owner']==student]
        if not incoming: st.info('No incoming requests for your listings yet.')
        for r in incoming:
            with st.container(border=True):
                st.markdown(f'**{r["title"]}** · Request from **{r["requester"]}** · `{r["status"]}`');st.write(r['message'] or 'No message.');b1,b2=st.columns(2)
                if b1.button('Accept request',key=f'accept_{r["id"]}'):
                    conn.execute("UPDATE requests SET status='Accepted' WHERE id=?",(r['id'],));conn.commit();st.success('Request accepted');st.rerun()
                if b2.button('Decline',key=f'decline_{r["id"]}'):
                    conn.execute("UPDATE requests SET status='Declined' WHERE id=?",(r['id'],));conn.commit();st.rerun()
    with tab2:
        mine=[r for r in reqs if r['requester']==student]
        if not mine: st.info('You have not sent any requests yet.')
        for r in mine:
            with st.container(border=True):
                st.markdown(f'**{r["title"]}** · Owner: {r["owner"]} · Status: `{r["status"]}`');st.write(r['message'] or '')
                if r['status']=='Accepted' and st.button('Mark exchange completed ♻️',key=f'complete_{r["id"]}'):
                    conn.execute("UPDATE listings SET status='Exchanged' WHERE id=?",(r['listing_id'],));conn.execute("UPDATE requests SET status='Completed' WHERE id=?",(r['id'],));conn.commit();st.success('Exchange marked complete!');st.rerun()

else:
    st.markdown('<div class="kicker">THE IDEA</div><div class="section-title">Good stuff deserves <span class="grad">another life.</span></div>',unsafe_allow_html=True)
    st.markdown('EcoSwap is a campus-focused circular economy platform. Students pass forward books, lab equipment, electronics, stationery and other useful items instead of letting them collect dust or buying new ones.')
    a,b,c=st.columns(3)
    for col,number,title,copy in zip([a,b,c],['01','02','03'],['Reuse','Connect','Impact'],['Extend the life of useful items.','Make campus exchanges simple.','Reduce waste and unnecessary purchases.']): col.markdown(f'<div class="metric"><div class="metric-value">{number}</div><b>{title}</b><p class="muted">{copy}</p></div>',unsafe_allow_html=True)
    st.info('Prototype note: this version stores data locally in SQLite on your computer. It is not yet an authenticated or cloud-hosted campus network. Meet in public campus locations and avoid sharing sensitive personal information.')

st.markdown('<div class="footer">ECOSWAP · KEEP GOOD THINGS MOVING · ♻️</div>',unsafe_allow_html=True)
