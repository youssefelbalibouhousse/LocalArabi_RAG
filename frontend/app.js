/* =========================================================================
   Logique du client — chatbot RAG
   =========================================================================
   Séparé de index.html volontairement : le balisage reste du balisage, et ce
   fichier est le seul à connaître les identifiants du DOM.

   Deux principes tenus dans tout le fichier :
     - tout contenu venu du serveur est écrit avec textContent, jamais
       innerHTML : la réponse d'un modèle n'est PAS une source de confiance ;
     - aucune classe CSS n'est construite par concaténation (le scanner de
       Tailwind ne verrait pas « bg-${couleur}-600 »).
   ========================================================================= */

(() => {
  'use strict';

  // API servie sur la MÊME origine que ce frontend (via le reverse proxy en
  // production). Une chaîne vide signifie « utilise l'origine courante ».
  const API_BASE = '';

  // --- Traductions -------------------------------------------------------
  // Toute chaîne affichée existe dans les DEUX langues.
  const I18N = {
    ar: {
      appTitle: 'مساعد البحث في المستندات',
      appSubtitle: 'الإجابة مع الاستشهاد بالمصدر: الملف والصفحة والأسطر',
      logout: 'خروج',
      changeLang: 'تغيير اللغة',
      loginTitle: 'تسجيل الدخول',
      username: 'اسم المستخدم',
      password: 'كلمة المرور',
      login: 'تسجيل الدخول',
      register: 'إنشاء حساب',
      registerRequired: 'يرجى إدخال اسم المستخدم وكلمة المرور.',
      inputPlaceholder: 'اكتب سؤالك…',
      send: 'إرسال',
      questionLabel: 'سؤالك',
      conversationLabel: 'المحادثة',
      loading: 'أبحث في المستندات…',
      processingError: 'تعذّر معالجة السؤال.',
      connectionError: 'تعذّر الاتصال بالخادم.',
      serverError: 'حدث خطأ في الخادم ولم يتمكن من معالجة الطلب.',
      sources: 'المصادر',
      excerpts: 'المقتطفات',
      welcome: (n) =>
        `مرحبًا ${n}. اطرح سؤالك حول المستندات، وستأتيك الإجابة مع مواضع الاستشهاد.`,
    },
    fr: {
      appTitle: 'Assistant de recherche documentaire',
      appSubtitle: 'Réponses citées : fichier, page et lignes',
      logout: 'Quitter',
      changeLang: 'Changer de langue',
      loginTitle: 'Connexion',
      username: "Nom d'utilisateur",
      password: 'Mot de passe',
      login: 'Se connecter',
      register: 'Créer un compte',
      registerRequired: "Veuillez saisir un nom d'utilisateur et un mot de passe.",
      inputPlaceholder: 'Écrivez votre question…',
      send: 'Envoyer',
      questionLabel: 'Votre question',
      conversationLabel: 'Conversation',
      loading: 'Recherche dans les documents…',
      processingError: 'La question n’a pas pu être traitée.',
      connectionError: 'Impossible de joindre le serveur.',
      serverError: 'Le serveur a rencontré une erreur et n’a pas pu traiter la demande.',
      sources: 'Sources',
      excerpts: 'Extraits cités',
      welcome: (n) =>
        `Bonjour ${n}. Posez votre question : la réponse viendra avec ses références.`,
    },
  };

  let currentLang = localStorage.getItem('chatbot_language') || null;

  const SVG_NS = 'http://www.w3.org/2000/svg';
  const CITE_MARKER = '📄';

  // --- Références DOM ----------------------------------------------------
  const byId = (id) => document.getElementById(id);

  const appTitle = byId('app-title');
  const appSubtitle = byId('app-subtitle');
  const logoutBtn = byId('logout-btn');
  const userBar = byId('user-bar');
  const userNameEl = byId('user-name');
  const langBackBtn = byId('lang-back-btn');

  const langPanel = byId('lang-panel');
  const langAr = byId('lang-ar');
  const langFr = byId('lang-fr');

  const authPanel = byId('auth-panel');
  const authForm = byId('auth-form');
  const authUsername = byId('auth-username');
  const authPassword = byId('auth-password');
  const authError = byId('auth-error');
  const loginBtn = byId('login-btn');
  const registerBtn = byId('register-btn');
  const loginTitle = byId('login-title');

  const chatWindow = byId('chat-window');
  const chatForm = byId('chat-form');
  const userInput = byId('user-input');
  const sendBtn = byId('send-btn');
  const composer = byId('composer');

  const t = (key) => I18N[currentLang][key];

  // =======================================================================
  // Citations
  // =======================================================================
  // Le backend ajoute les sources et les extraits À LA SUITE du texte de la
  // réponse (un seul champ `answer`). On les en sépare pour leur donner une
  // présentation digne : dans une édition savante, la référence n'est pas une
  // note grise reléguée en bas, c'est ce qui rend l'affirmation vérifiable.
  //
  // ⚠️ Ce découpage s'appuie sur le TEXTE produit par le backend. Il est donc
  // volontairement défensif : au moindre doute, on affiche la réponse entière
  // plutôt que de risquer d'en perdre un fragment.
  //
  // Le correctif durable est côté API : renvoyer `answer` (la prose seule) plus
  // `sources` et `excerpts` en champs structurés — la réponse les contient déjà.

  function valueOfLabel(line) {
    // « 📄 Sources : a.pdf — page 5 »  ->  « a.pdf — page 5 »
    const withoutEmoji = line.replace(/^📄\s*/, '');
    const match = withoutEmoji.match(/^[^:：]*[:：]\s*([\s\S]*)$/);
    return (match ? match[1] : withoutEmoji).trim();
  }

  function parseExcerpt(line, index) {
    // « 1. « texte » — a.pdf — page 5 (lignes 1-9) »
    const withoutNumber = line.replace(/^\d+\.\s*/, '');
    // Quantificateur GLOUTON : on veut le DERNIER « » », pas le premier —
    // l'extrait cité peut lui-même contenir des guillemets.
    const match = withoutNumber.match(/^«\s*([\s\S]*)»\s*—\s*([\s\S]+)$/);
    if (!match) return { text: withoutNumber || line, ref: '', index };
    return { text: match[1].trim(), ref: match[2].trim(), index };
  }

  function parseAnswer(raw) {
    const text = raw ?? '';
    const at = text.indexOf(CITE_MARKER);
    if (at === -1) return { prose: text.trim(), refs: '', excerpts: [] };

    const lines = text
      .slice(at)
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean);

    return {
      prose: text.slice(0, at).trim(),
      refs: valueOfLabel(lines[0] ?? ''),
      excerpts: lines
        .slice(1)
        .filter((line) => !line.startsWith(CITE_MARKER))
        .map(parseExcerpt),
    };
  }

  /** Petite étoile à huit branches : le motif géométrique du projet. */
  function starMark() {
    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('viewBox', '0 0 14 14');
    svg.setAttribute('class', 'citation__mark');
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('focusable', 'false');

    // Un carré et le même pivoté de 45° : la construction la plus simple d'une
    // étoile à huit branches. Géométrique, donc austère.
    for (const rotation of [null, 'rotate(45 7 7)']) {
      const rect = document.createElementNS(SVG_NS, 'rect');
      rect.setAttribute('x', '3.5');
      rect.setAttribute('y', '3.5');
      rect.setAttribute('width', '7');
      rect.setAttribute('height', '7');
      rect.setAttribute('fill', 'none');
      rect.setAttribute('stroke', 'currentColor');
      rect.setAttribute('stroke-width', '1');
      if (rotation) rect.setAttribute('transform', rotation);
      svg.appendChild(rect);
    }
    return svg;
  }

  function citationLabel(text, withMark) {
    const p = document.createElement('p');
    p.className = withMark ? 'citation__label' : 'citation__label citation__label--sub';
    if (withMark) p.appendChild(starMark());
    p.appendChild(document.createTextNode(text));
    return p;
  }

  function buildCitation({ refs, excerpts }) {
    const block = document.createElement('div');
    block.className = 'citation';

    if (refs) {
      block.appendChild(citationLabel(t('sources'), true));
      const line = document.createElement('p');
      line.className = 'citation__refs';
      line.textContent = refs;
      block.appendChild(line);
    }

    if (excerpts.length) {
      block.appendChild(citationLabel(t('excerpts'), false));

      const list = document.createElement('ul');
      list.className = 'citation__list';

      for (const item of excerpts) {
        const li = document.createElement('li');
        li.className = 'citation__item';
        // La voix Naskh est posée par le CSS : ici, le texte cité.
        li.textContent = item.text;

        if (item.ref) {
          const ref = document.createElement('cite');
          ref.className = 'citation__source';
          ref.textContent = item.ref;
          li.appendChild(ref);
        }
        list.appendChild(li);
      }
      block.appendChild(list);
    }

    return block.childElementCount ? block : null;
  }

  // =======================================================================
  // Conversation
  // =======================================================================

  function scrollToLatest() {
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  function newTurn(bubbleClasses) {
    const turn = document.createElement('div');
    turn.className = 'turn';

    const bubble = document.createElement('div');
    bubble.className = bubbleClasses;

    turn.appendChild(bubble);
    chatWindow.appendChild(turn);
    return bubble;
  }

  function appendMessage(text, isUser = false, citations = null) {
    const bubble = newTurn(isUser ? 'bubble bubble--user' : 'bubble bubble--assistant');

    const body = document.createElement('p');
    body.className = 'bubble__text';
    body.textContent = text;
    bubble.appendChild(body);

    if (citations) {
      const block = buildCitation(citations);
      if (block) bubble.appendChild(block);
    }

    scrollToLatest();
    return bubble;
  }

  function appendAnswer(raw) {
    const parsed = parseAnswer(raw);
    // Format inattendu : on montre la réponse telle quelle plutôt que rien.
    if (!parsed.prose) {
      appendMessage(raw, false);
      return;
    }
    appendMessage(parsed.prose, false, parsed);
  }

  /**
   * L'accueil n'est pas une RÉPONSE : il ne doit donc pas prendre la forme
   * d'une carte de réponse pleine largeur, mais celle d'une note discrète.
   */
  function appendWelcome(name) {
    const bubble = newTurn('bubble bubble--note');

    const body = document.createElement('p');
    body.className = 'bubble__text';
    body.textContent = I18N[currentLang].welcome(name);
    bubble.appendChild(body);

    scrollToLatest();
  }

  /** Indicateur d'attente. Renvoie l'élément à retirer ensuite. */
  function appendWaiting() {
    const bubble = newTurn('bubble bubble--assistant');

    const row = document.createElement('div');
    row.className = 'waiting';

    const spinner = document.createElement('span');
    spinner.className = 'spinner';
    spinner.setAttribute('aria-hidden', 'true');

    const label = document.createElement('span');
    label.textContent = t('loading');

    row.append(spinner, label);
    bubble.appendChild(row);
    scrollToLatest();

    return bubble.closest('.turn');
  }

  // =======================================================================
  // Langue et affichage des vues
  // =======================================================================

  function applyLang() {
    const isRtl = currentLang === 'ar';

    document.documentElement.lang = currentLang;
    document.documentElement.dir = isRtl ? 'rtl' : 'ltr';
    document.title = I18N[currentLang].appTitle;

    appTitle.textContent = I18N[currentLang].appTitle;
    appSubtitle.textContent = I18N[currentLang].appSubtitle;
    logoutBtn.textContent = I18N[currentLang].logout;
    loginTitle.textContent = I18N[currentLang].loginTitle;
    authUsername.placeholder = I18N[currentLang].username;
    authPassword.placeholder = I18N[currentLang].password;
    loginBtn.textContent = I18N[currentLang].login;
    registerBtn.textContent = I18N[currentLang].register;
    userInput.placeholder = I18N[currentLang].inputPlaceholder;
    sendBtn.textContent = I18N[currentLang].send;

    // Les libellés ACCESSIBLES aussi doivent suivre la langue : un lecteur
    // d'écran en français ne doit pas annoncer « سؤالك ».
    userInput.setAttribute('aria-label', I18N[currentLang].questionLabel);
    chatWindow.setAttribute('aria-label', I18N[currentLang].conversationLabel);
    authUsername.setAttribute('aria-label', I18N[currentLang].username);
    authPassword.setAttribute('aria-label', I18N[currentLang].password);

    // Bouton en icône : le libellé visible est un glyphe, le texte part donc
    // dans les attributs accessibles plutôt que dans le contenu.
    langBackBtn.setAttribute('aria-label', I18N[currentLang].changeLang);
    langBackBtn.setAttribute('title', I18N[currentLang].changeLang);

    // Plus AUCUN attribut de direction posé à la main sur les champs : les
    // propriétés logiques du CSS (text-align: start, padding-inline) suivent
    // la direction du document. Quinze lignes de JS supprimées.
  }

  function show(panel) {
    langPanel.classList.toggle('hidden', panel !== 'lang');
    authPanel.classList.toggle('hidden', panel !== 'auth');
    chatWindow.classList.toggle('hidden', panel !== 'chat');
    composer.classList.toggle('hidden', panel !== 'chat');
    userBar.classList.toggle('hidden', panel !== 'chat');
    langBackBtn.classList.toggle('hidden', panel === 'lang');
  }

  function showLangPanel() {
    chatWindow.replaceChildren();
    show('lang');
  }

  function chooseLang(lang) {
    currentLang = lang;
    localStorage.setItem('chatbot_language', lang);
    applyLang();
    initAfterLang();
  }

  // --- Jeton d'authentification ------------------------------------------
  const getToken = () => localStorage.getItem('chatbot_token');
  const setToken = (token) => localStorage.setItem('chatbot_token', token);
  const clearToken = () => localStorage.removeItem('chatbot_token');

  function showAuth() {
    clearToken();
    show('auth');
  }

  function showChat(username) {
    userNameEl.textContent = username;
    show('chat');
    userInput.focus();
  }

  function showAuthError(message) {
    authError.textContent = message;
    authError.classList.remove('hidden');
  }

  function hideAuthError() {
    authError.classList.add('hidden');
  }

  /** Enveloppe fetch : ajoute le jeton et gère automatiquement le 401. */
  async function apiFetch(path, options = {}) {
    const headers = { ...(options.headers || {}) };
    const token = getToken();
    if (token) headers.Authorization = `Bearer ${token}`;

    const response = await fetch(API_BASE + path, { ...options, headers });

    if (response.status === 401) {
      showAuth();
      throw new Error('unauthorized');
    }
    return response;
  }

  /**
   * Lit le corps d'une réponse SANS supposer qu'il est en JSON.
   *
   * Une exception non gérée côté serveur renvoie « Internal Server Error » en
   * TEXTE BRUT. Appeler `response.json()` dessus lève donc une erreur, qui est
   * ensuite présentée à l'utilisateur comme une panne réseau — alors que le
   * serveur a bel et bien répondu. Un diagnostic faux, et c'est exactement le
   * genre d'erreur qui coûte des heures de recherche.
   */
  async function readBody(response) {
    const brut = await response.text();
    try {
      return JSON.parse(brut);
    } catch {
      return { detail: brut.trim() };
    }
  }

  // --- Connexion / inscription -------------------------------------------
  async function login(username, password) {
    const response = await fetch(`${API_BASE}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({ username, password }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || t('processingError'));
    setToken(data.access_token);
  }

  async function register(username, password) {
    const response = await fetch(`${API_BASE}/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || t('processingError'));
    return data.username;
  }

  authForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    hideAuthError();

    const username = authUsername.value.trim();
    const password = authPassword.value;
    if (!username || !password) return;

    try {
      await login(username, password);
      authForm.reset();
      showChat(username);
    } catch (err) {
      showAuthError(err.message);
    }
  });

  registerBtn.addEventListener('click', async () => {
    hideAuthError();

    const username = authUsername.value.trim();
    const password = authPassword.value;
    if (!username || !password) {
      showAuthError(t('registerRequired'));
      return;
    }

    try {
      await register(username, password);
      await login(username, password);
      authForm.reset();
      showChat(username);
    } catch (err) {
      showAuthError(err.message);
    }
  });

  logoutBtn.addEventListener('click', () => {
    chatWindow.replaceChildren();
    showAuth();
  });

  // --- Envoi d'une question ----------------------------------------------
  chatForm.addEventListener('submit', async (event) => {
    event.preventDefault();

    const question = userInput.value.trim();
    if (!question) return;

    appendMessage(question, true);
    userInput.value = '';

    // Désactiver l'envoi évite la double soumission pendant l'attente — c'est
    // la seule animation qu'on s'autorise ici : un changement d'état, pas un
    // effet. Envoyer un message est une action à 100+ par jour : aucune
    // animation ne doit s'y ajouter.
    sendBtn.disabled = true;
    const waiting = appendWaiting();

    try {
      const response = await apiFetch(
        `/ask?question=${encodeURIComponent(question)}&language=${currentLang}`,
      );
      const data = await readBody(response);

      waiting.remove();

      if (response.ok) {
        appendAnswer(data.answer);
      } else if (response.status >= 500) {
        // Le serveur a RÉPONDU, mais en échec (modèle Ollama manquant, service
        // injoignable...). L'annoncer comme une panne réseau enverrait
        // l'utilisateur — et le développeur — sur une fausse piste.
        appendMessage(t('serverError'), false);
        console.error('Erreur serveur', response.status, ':', data.detail);
      } else {
        appendMessage(data.detail || t('processingError'), false);
      }
    } catch (error) {
      waiting.remove();
      if (error.message !== 'unauthorized') appendMessage(t('connectionError'), false);
      console.error('Erreur de connexion au backend :', error);
    } finally {
      sendBtn.disabled = false;
      userInput.focus();
    }
  });

  // --- Démarrage ---------------------------------------------------------
  function initAfterLang() {
    if (!getToken()) {
      showAuth();
      return;
    }

    apiFetch('/me')
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => {
        if (!data) {
          showAuth();
          return;
        }
        showChat(data.username);
        appendWelcome(data.username);
      })
      .catch(() => showAuth());
  }

  /** Configuration publique (non sensible) : sert à masquer l'inscription. */
  async function loadPublicConfig() {
    try {
      const response = await fetch(`${API_BASE}/config`);
      if (!response.ok) return;

      const data = await response.json();
      // Inscription fermée (pilote) : on masque le bouton de création.
      if (data.allow_registration === false) registerBtn.classList.add('hidden');
    } catch (err) {
      // En cas d'échec on laisse le bouton visible : le backend refusera de
      // toute façon si l'inscription est fermée.
      console.error('Configuration publique indisponible :', err);
    }
  }

  async function bootstrap() {
    await loadPublicConfig();

    if (currentLang) {
      applyLang();
      initAfterLang();
      return;
    }

    show('lang');
  }

  langAr.addEventListener('click', () => chooseLang('ar'));
  langFr.addEventListener('click', () => chooseLang('fr'));
  langBackBtn.addEventListener('click', showLangPanel);

  bootstrap();
})();
