from client.templates import template
import reflex as rx

class AnalyzeState(rx.State):
    is_healthy: str
    gemini: str
    nerf_url: str
    file_name: str

    async def update_is_healthy(self, new_is_healthy):
        self.is_healthy = new_is_healthy
        
    async def update_gemini(self, new_gemini):
        self.gemini = new_gemini
    
    async def update_nerf_url(self, new_nerf_url):
        self.nerf_url = new_nerf_url
    
    async def update_file_name(self, new_file_name):
        self.file_name = new_file_name
    
@template(route="/analyze", title="Analyze")
def analyze() -> rx.Component:
    return rx.vstack(
        rx.link(
            rx.button(
                "Back",
            ),
            href="/"
        ),
        rx.heading("Results", size="8"),
        #rx.text(AnalyzeState.is_healthy),
        rx.text(AnalyzeState.gemini),
        rx.video(
            url=AnalyzeState.nerf_url,
            width="400px",
            height="auto",
            playing=True,
            loop=True
        ),
    )
