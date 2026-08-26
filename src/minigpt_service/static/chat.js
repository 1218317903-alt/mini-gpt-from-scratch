const state = { conversationId: null, runId: null, controller: null };
const $ = (selector) => document.querySelector(selector);

async function request(url, options = {}) {
  const response = await fetch(url, { headers: { "content-type": "application/json", ...(options.headers || {}) }, ...options });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `HTTP ${response.status}`);
  }
  return response.status === 204 ? null : response.json();
}

function messageNode(message) {
  const node = document.createElement("article");
  node.className = `message ${message.role} ${message.status === "error" ? "error" : ""}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = message.content;
  const meta = document.createElement("span");
  meta.className = "meta";
  meta.textContent = `${message.status}${message.generated_tokens != null ? ` · ${message.generated_tokens} tokens` : ""}`;
  bubble.append(meta);
  node.append(bubble);
  return node;
}

function renderConversation(conversation) {
  state.conversationId = conversation.id;
  $("#conversation-title").value = conversation.title;
  $("#summary").textContent = conversation.summary || "尚未触发上下文压缩。";
  const container = $("#messages");
  container.replaceChildren();
  if (!conversation.messages.length) {
    container.innerHTML = '<div class="empty-state"><h1>开始一段本地对话</h1><p>历史会自动保存，较早轮次会按上下文预算压缩。</p></div>';
  } else {
    conversation.messages.filter((m) => m.content).forEach((m) => container.append(messageNode(m)));
  }
  container.scrollTop = container.scrollHeight;
  document.querySelectorAll(".conversation").forEach((item) => item.classList.toggle("active", item.dataset.id === conversation.id));
}

async function loadConversations(selectFirst = false) {
  const payload = await request("/conversations");
  const list = $("#conversation-list");
  list.replaceChildren();
  payload.conversations.forEach((conversation) => {
    const button = document.createElement("button");
    button.className = "conversation";
    button.dataset.id = conversation.id;
    button.textContent = conversation.title;
    button.onclick = () => openConversation(conversation.id);
    list.append(button);
  });
  if (selectFirst && payload.conversations.length) await openConversation(payload.conversations[0].id);
}

async function openConversation(id) {
  renderConversation(await request(`/conversations/${id}`));
}

async function newConversation() {
  const conversation = await request("/conversations", { method: "POST", body: JSON.stringify({ title: "新对话" }) });
  await loadConversations();
  renderConversation(conversation);
  $("#prompt").focus();
}

function setGenerating(value) {
  $("#send").disabled = value;
  $("#stop").disabled = !value;
  $("#prompt").disabled = value;
  $("#status").textContent = value ? "正在生成…" : "就绪";
}

function parseSSEBlock(block) {
  const event = block.match(/^event:\s*(.+)$/m)?.[1];
  const data = block.match(/^data:\s*(.+)$/m)?.[1];
  return event && data ? { event, data: JSON.parse(data) } : null;
}

async function sendMessage(content) {
  if (!state.conversationId) await newConversation();
  setGenerating(true);
  state.controller = new AbortController();
  let assistantBubble = null;
  try {
    const response = await fetch(`/conversations/${state.conversationId}/messages/stream`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ content, client_request_id: crypto.randomUUID(), max_new_tokens: 512 }),
      signal: state.controller.signal,
    });
    if (!response.ok) throw new Error((await response.json()).detail || `HTTP ${response.status}`);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop();
      for (const block of blocks) {
        const item = parseSSEBlock(block);
        if (!item) continue;
        if (item.event === "start") {
          state.runId = item.data.run_id;
          $("#run-id").textContent = state.runId;
          $("#context-tokens").textContent = item.data.estimated_context_tokens;
          $("#omitted-count").textContent = item.data.omitted_messages;
          const messages = $("#messages");
          messages.querySelector(".empty-state")?.remove();
          messages.append(messageNode(item.data.user_message));
          assistantBubble = messageNode(item.data.assistant_message);
          messages.append(assistantBubble);
        } else if (item.event === "delta") {
          const bubble = assistantBubble.querySelector(".bubble");
          const meta = bubble.querySelector(".meta");
          bubble.insertBefore(document.createTextNode(item.data.text), meta);
          $("#messages").scrollTop = $("#messages").scrollHeight;
        } else if (item.event === "done") {
          $("#prompt-tokens").textContent = item.data.message.prompt_tokens ?? "—";
          $("#generated-tokens").textContent = item.data.message.generated_tokens ?? "—";
          $("#finish-reason").textContent = item.data.message.finish_reason ?? "—";
        } else if (item.event === "error") {
          throw new Error(item.data.detail);
        }
      }
    }
  } finally {
    state.runId = null;
    state.controller = null;
    setGenerating(false);
    await loadConversations();
    if (state.conversationId) await openConversation(state.conversationId);
  }
}

$("#new-chat").onclick = newConversation;
$("#delete-chat").onclick = async () => {
  if (!state.conversationId || !confirm("删除这个会话？")) return;
  await request(`/conversations/${state.conversationId}`, { method: "DELETE" });
  state.conversationId = null;
  await loadConversations(true);
};
$("#conversation-title").onchange = async (event) => {
  if (!state.conversationId) return;
  await request(`/conversations/${state.conversationId}`, { method: "PATCH", body: JSON.stringify({ title: event.target.value }) });
  await loadConversations();
};
$("#stop").onclick = async () => {
  if (state.runId) await request(`/conversations/runs/${state.runId}/cancel`, { method: "POST" }).catch(() => null);
};
$("#composer").onsubmit = async (event) => {
  event.preventDefault();
  const prompt = $("#prompt");
  const content = prompt.value.trim();
  if (!content) return;
  prompt.value = "";
  try { await sendMessage(content); } catch (error) { $("#status").textContent = error.message; setGenerating(false); }
};
$("#prompt").onkeydown = (event) => {
  if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); $("#composer").requestSubmit(); }
};

loadConversations(true).then(async () => { if (!state.conversationId) await newConversation(); }).catch((error) => { $("#status").textContent = error.message; });
