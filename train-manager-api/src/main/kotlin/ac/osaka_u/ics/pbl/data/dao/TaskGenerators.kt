package ac.osaka_u.ics.pbl.data.dao

import ac.osaka_u.ics.pbl.common.ParameterMapSerializer
import ac.osaka_u.ics.pbl.common.TaskType
import org.jetbrains.exposed.dao.id.IntIdTable
import org.jetbrains.exposed.sql.json.jsonb

object TaskGenerators : IntIdTable("task_generators") {
    val type = enumeration("type", TaskType::class)
    val name = varchar("name", 255)
    val weight = integer("weight").default(1)
    val parameters = jsonb<Map<String,Any>>("parameters", format, ParameterMapSerializer)
}