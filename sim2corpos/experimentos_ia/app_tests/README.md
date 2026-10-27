Testes pelos aplicativos (modelo + ferramenta de execução), feitos em julho de 2026 (confirmado pelo Rafael em 23/10) e colados no chat em 22/10/2026.
Mesmo enunciado do ensaio de julho (mundo_01/prompt.txt): tabela t=0..17, sem exigir programa.
- opus-4.8_app-desktop.csv: Claude Opus 4.8 pelo app desktop (Mac). Escreveu RK4 próprio com dt=1e-5 e rodou via bash. Verificou centro de massa.
- haiku-4.5_app.csv: Claude Haiku 4.5 pelo app. Usou scipy solve_ivp RK45 com tolerância padrão, t_eval inteiro.
Erros contra a solução analítica: ver erros.txt (gerado por calc_erro.py).
