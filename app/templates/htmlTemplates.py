css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://static.vecteezy.com/system/resources/previews/009/971/217/non_2x/chat-bot-icon-isolated-contour-symbol-illustration-vector.jpg" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''



user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.pinimg.com/564x/ce/b8/5c/ceb85c4c40dc6fac6eb6cad0c5bc5fe3.jpg" alt="Laude" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover; ">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''