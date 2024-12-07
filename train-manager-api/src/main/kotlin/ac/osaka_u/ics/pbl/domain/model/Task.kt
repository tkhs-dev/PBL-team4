package ac.osaka_u.ics.pbl.domain.model

import ac.osaka_u.ics.pbl.common.TaskType
import java.util.*

data class Task(
    val id: UUID,
    val completed: Boolean,
    val baseModelId: UUID?,
    val type: TaskType,
    val parameter: Map<String, Any>
)
