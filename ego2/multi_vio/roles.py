"""Body roles and the reference policy shared by GUI and runtime validation."""

ROLES = ('chest', 'head', 'left_hand', 'right_hand')
LABELS = dict(chest='胸部', head='头部', left_hand='左手', right_hand='右手')
# Rerun's bundled UI font lacks CJK glyphs; keep the setup panel Chinese.
VIEWER_LABELS = dict(chest='Chest', head='Head', left_hand='Left hand', right_hand='Right hand')


def reference_for(roles):
    """A sole camera references itself; multiple cameras require chest."""
    roles = tuple(roles)
    if not roles:
        raise ValueError('请至少选择一台相机。')
    if len(roles) != len(set(roles)) or not set(roles) <= set(ROLES):
        raise ValueError('部位分配格式无效')
    if len(roles) == 1:
        return roles[0]
    if 'chest' not in roles:
        raise ValueError('未选择胸部时，仅支持单相机运行；启用多相机请先为胸部分配相机。')
    return 'chest'
