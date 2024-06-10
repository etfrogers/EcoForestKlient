package com.etfrogers.ecoforestklient

import kotlinx.datetime.TimeZone
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals



internal class ChunkTest {

//    private val testSample: Sample = Sample()

    @Test
    fun `test chunks 01-04-2024`() {
        val str = File("2024-04-01.csv").readText()
        val (timestamps, data) = EcoForestClient.processFileData(str, timezone = TimeZone.UTC)
        val day = DayData(TimeZone.UTC, timestamps, data)
        val firstNonZero = day.consumption.first { it > 0 }
        val chunks = day.chunks()
        assertEquals(firstNonZero, chunks[0].consumption[0])

        assertEquals(2, chunks.size)
        assertEquals(4, chunks[0].size)
        assertEquals(7, chunks[1].size)

        assertEquals(chunks[0].type, ChunkClass.HEATING)
        assertEquals(chunks[1].type, ChunkClass.SOLAR_HEATING)
        val lastNonZero = day.consumption.last { it > 0 }
        assertEquals(lastNonZero, chunks.last().consumption.last())
    }

    @Test
    fun `test chunks 11-03-2024`() {
        val str = File("2024-03-11.csv").readText()
        val (timestamps, data) = EcoForestClient.processFileData(str, timezone = TimeZone.UTC)
        val day = DayData(TimeZone.UTC, timestamps, data)
        val chunks = day.chunks()
        assertEquals(10, chunks.size)
        chunks.dropLast(1).forEach {
            assertEquals(ChunkClass.HEATING, it.type)
        }
        assertEquals(ChunkClass.DHW, chunks.last().type)
    }

    @Test
    fun `test chunks 13-03-2024`() {
        val str = File("2024-03-13.csv").readText()
        val (timestamps, data) = EcoForestClient.processFileData(str, timezone = TimeZone.UTC)
        val day = DayData(TimeZone.UTC, timestamps, data)
        val chunks = day.chunks()
        assertEquals(7, chunks.size)

        assertEquals(ChunkClass.DHW, chunks.first().type)
        assertEquals(ChunkClass.LEGIONNAIRES, chunks[5].type)
        assertEquals(ChunkClass.HEATING, chunks.last().type)
    }

    @Test
    fun `test chunks 14-03-2024`() {
        val str = File("2024-03-14.csv").readText()
        val (timestamps, data) = EcoForestClient.processFileData(str, timezone = TimeZone.UTC)
        val day = DayData(TimeZone.UTC, timestamps, data)
        val chunks = day.chunks()
        assertEquals(6, chunks.size)

        assertEquals(ChunkClass.HEATING, chunks[0].type)
        assertEquals(ChunkClass.HEATING, chunks[1].type)
        assertEquals(ChunkClass.HEATING, chunks[2].type)
        assertEquals(ChunkClass.SOLAR_HEATING, chunks[3].type)
        assertEquals(ChunkClass.UNKNOWN, chunks[4].type)
        assertEquals(ChunkClass.HEATING, chunks[5].type)
    }
}