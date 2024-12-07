package ac.osaka_u.ics.pbl.domain.model

import ac.osaka_u.ics.pbl.common.AssignmentStatus
import kotlinx.datetime.Instant
import java.util.*

data class Assignment(
    val id: UUID,
    val assignedAt: Instant,
    val clientId: Int,
    val deadline: Instant,
    val status: AssignmentStatus,
    val statusChangedAt: Instant,
    val task: Task,
)
