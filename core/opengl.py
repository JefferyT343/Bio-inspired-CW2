try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
except Exception:  # pragma: no cover - fallback for headless runs
    def _noop(*_args, **_kwargs):
        return None

    # Common GL constants used by this project
    GL_POLYGON = 0
    GL_TRIANGLE_FAN = 0
    GL_LINES = 0
    GL_LINE_STRIP = 0
    GL_BLEND = 0
    GLU_FILL = 0

    # GL functions (no-op in headless mode)
    glBegin = _noop
    glEnd = _noop
    glColor4f = _noop
    glColor4fv = _noop
    glVertex2d = _noop
    glLineWidth = _noop
    glEnable = _noop
    glDisable = _noop
    glPushMatrix = _noop
    glPopMatrix = _noop
    glTranslated = _noop
    glRotated = _noop
    glScaled = _noop
    glFinish = _noop

    # GLU functions (no-op in headless mode)
    gluNewQuadric = _noop
    gluQuadricDrawStyle = _noop
    gluDisk = _noop
    gluDeleteQuadric = _noop
